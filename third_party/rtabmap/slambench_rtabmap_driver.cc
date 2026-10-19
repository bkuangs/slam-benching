// Headless offline driver for RTAB-Map (stereo) used by slam-benching.
//
// Runs RTAB-Map's own stereo visual odometry (rtabmap::Odometry) and graph SLAM
// (rtabmap::Rtabmap) in one process on the slambench offline input format, the way
// rtabmap_ros' stereo_odometry + rtabmap nodes would, but deterministic with respect to the
// data: the SLAM update rate is applied on data timestamps, not wall time.
//
// Usage:
//   rtabmap_driver <params.txt> <input_dir> <output_dir>
//                  --camera fx,fy,cx,cy,baseline,width,height
//                  [--detection-rate HZ] [--playback-rate R] [--log-features] [--localize]
//                  [--threaded-map-update]
//   rtabmap_driver --print-capabilities
//
//   params.txt         one RTAB-Map parameter per line, "Group/Name=value" ('#' comments).
//   --camera           rectified stereo pinhole model of the input images (baseline in m).
//   --detection-rate   SLAM (map) update rate in Hz on data time; 0 = every frame with odometry.
//                      Mirrors Rtabmap/DetectionRate of the ROS / standalone apps.
//   --playback-rate R  R > 0 throttles input to R x the recorded frame rate (1.0 = real time);
//                      R = 0 feeds frames as fast as possible.
//   --log-features     write features.bin (odometry keypoints of every frame and their status).
//   --localize         map reuse: output_dir/rtabmap.db must already hold a saved map (a copy; it
//                      is never saved back) and params.txt must set Mem/IncrementalMemory=false.
//                      RTAB-Map localizes against the map instead of extending it; nodes.csv and
//                      links.csv are not written, map_loaded_nodes.csv lists the loaded map.
//   --threaded-map-update  run map updates (Rtabmap::process) on a worker thread, as the separate
//                      odometry and rtabmap nodes of rtabmap_ros do, instead of after the frame's
//                      odometry on the same thread. Odometry never waits for a map update; the
//                      online pose uses the latest finished update's map correction; an update
//                      request still waiting when the next is due is dropped (latest wins).
//                      track_time_s is then the odometry time only.
//
// Input (input_dir):
//   frames.csv : timestamp_ns,left_path,right_path   (rectified stereo pair)
//
// Output (output_dir):
//   frame_log.csv       per-frame odometry/SLAM diagnostics and the online cam0 pose
//                       (map correction x odometry, i.e. what a robot would use at that moment)
//                       plus the raw cam0 odometry pose; flushed every frame.
//                       coord_session numbers the world frame of the online pose: it changes when
//                       the root (oldest node) of RTAB-Map's optimized graph changes, e.g. after a
//                       new map is started or maps are joined.
//                       localized = 1 from the first accepted loop closure / proximity link to a
//                       map node on (sticky; before it the pose is only RTAB-Map's prior: the last
//                       pose saved in the map); loc_event = 1 on updates that accepted one;
//                       loc_rejected = 1 on updates whose loop hypothesis was rejected;
//                       loc_correction_m = norm of the odometry correction that update applied.
//   map_loaded_nodes.csv  (--localize) optimized cam0 pose of every map node at load time.
//   nodes.csv           every node added to the map: id, map id, stamp, cam0 odometry pose and
//                       (if in the final optimized graph) the final optimized cam0 pose.
//   links.csv           constraints of the final graph (from, to, type).
//   rtabmap.db          RTAB-Map database (localization map: images, features, graph, local grids).
//   features.bin        (--log-features) little-endian records after the 8-byte magic "SBFEAT01":
//                       int64 timestamp_ns, int32 frame_idx, int32 n, then n x
//                       {float32 x, float32 y, uint8 status}; x,y = keypoint in the left image;
//                       status 0 = not matched to the odometry local map, 1 = inlier, 2 = matched
//                       but rejected by the motion estimate.
//   driver_status.json  exit status written by the driver itself.
//   driver_timing.json  frame-loop wall/CPU time, odometry vs map-update CPU, dropped updates.
//
// Poses are written for the left camera optical frame (cam0): RTAB-Map's base frame pose times
// the camera local transform (the default optical rotation, so RTAB-Map's base frame is x-forward,
// z-up relative to the first camera pose, which its ground segmentation assumes).

#include <rtabmap/core/Link.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/Statistics.h>
#include <rtabmap/core/StereoCameraModel.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/utilite/ULogger.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <sys/resource.h>
#include <time.h>
#include <vector>

using namespace std;
using rtabmap::Transform;

namespace {

struct FrameEntry {
    long long t_ns;
    string left;
    string right;
};

struct NodeRecord {
    int id;
    int map_id;
    long long t_ns;
    Transform odom_cam;
};

vector<string> Split(const string &line, char sep) {
    vector<string> out;
    string cell;
    stringstream ss(line);
    while (getline(ss, cell, sep)) out.push_back(cell);
    if (!line.empty() && line.back() == sep) out.push_back("");
    return out;
}

string Trim(const string &s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool LoadFrames(const string &path, vector<FrameEntry> &frames) {
    ifstream f(path);
    if (!f.is_open()) return false;
    string line;
    getline(f, line);  // header
    while (getline(f, line)) {
        if (line.empty()) continue;
        auto c = Split(line, ',');
        if (c.size() < 3 || c[2].empty()) return false;
        frames.push_back({stoll(c[0]), c[1], c[2]});
    }
    return true;
}

bool LoadParams(const string &path, rtabmap::ParametersMap &params, string &err) {
    ifstream f(path);
    if (!f.is_open()) { err = "cannot open " + path; return false; }
    const rtabmap::ParametersMap &defaults = rtabmap::Parameters::getDefaultParameters();
    string line;
    while (getline(f, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto eq = line.find('=');
        if (eq == string::npos) { err = "malformed parameter line: " + line; return false; }
        const string key = Trim(line.substr(0, eq)), value = Trim(line.substr(eq + 1));
        if (defaults.find(key) == defaults.end()) { err = "unknown RTAB-Map parameter: " + key; return false; }
        params[key] = value;
    }
    return true;
}

string JsonEscape(const string &s) {
    string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

void WriteStatus(const string &path, const string &status, const string &message,
                 size_t n_frames_total, size_t n_frames_fed, size_t n_slam_updates = 0, size_t n_nodes = 0) {
    ofstream f(path);
    f << "{\n  \"status\": \"" << status << "\",\n  \"message\": \"" << JsonEscape(message)
      << "\",\n  \"n_frames_total\": " << n_frames_total
      << ",\n  \"n_frames_fed\": " << n_frames_fed
      << ",\n  \"n_slam_updates\": " << n_slam_updates
      << ",\n  \"n_nodes_added\": " << n_nodes << "\n}\n";
}

void WritePose(ostream &o, const Transform &T) {
    if (T.isNull()) { o << ",,,,,,"; return; }
    const Eigen::Quaternionf q = T.getQuaternionf();
    o << setprecision(9) << T.x() << "," << T.y() << "," << T.z() << ","
      << q.x() << "," << q.y() << "," << q.z() << "," << q.w();
}

double ThreadCpuSeconds() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double ProcessCpuSeconds() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

int GraphRoot(const map<int, Transform> &poses) {
    for (const auto &kv : poses) if (kv.first > 0) return kv.first;  // ids ascend; <=0 are landmarks
    return 0;
}

void PrintCapabilities() {
    cout << "rtabmap_version=" << rtabmap::Parameters::getVersion() << "\n";
    const rtabmap::ParametersMap &d = rtabmap::Parameters::getDefaultParameters();
    cout << "optimizer_default_strategy=" << d.at(rtabmap::Parameters::kOptimizerStrategy()) << "\n";
    cout << "optimizer_g2o=" << rtabmap::Optimizer::isAvailable(rtabmap::Optimizer::kTypeG2O) << "\n";
    cout << "optimizer_gtsam=" << rtabmap::Optimizer::isAvailable(rtabmap::Optimizer::kTypeGTSAM) << "\n";
    cout << "optimizer_toro=" << rtabmap::Optimizer::isAvailable(rtabmap::Optimizer::kTypeTORO) << "\n";
    cout << "grid_3d_default=" << d.at(rtabmap::Parameters::kGrid3D()) << "\n";
    cout << "threaded_map_update=1\n";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && string(argv[1]) == "--print-capabilities") {
        PrintCapabilities();
        return 0;
    }
    if (argc < 4) {
        cerr << "Usage: rtabmap_driver <params.txt> <input_dir> <output_dir> --camera fx,fy,cx,cy,b,w,h"
                " [--detection-rate HZ] [--playback-rate R] [--log-features] [--localize]"
                " [--threaded-map-update]" << endl;
        return 2;
    }
    const string params_path = argv[1], input_dir = argv[2], output_dir = argv[3];
    const string status_path = output_dir + "/driver_status.json";
    double playback_rate = 0.0, detection_rate = 1.0;
    bool log_features = false, localize = false, threaded = false;
    vector<double> cam;
    for (int i = 4; i < argc; ++i) {
        string a = argv[i];
        if (a == "--playback-rate" && i + 1 < argc) playback_rate = atof(argv[++i]);
        else if (a == "--detection-rate" && i + 1 < argc) detection_rate = atof(argv[++i]);
        else if (a == "--log-features") log_features = true;
        else if (a == "--localize") localize = true;
        else if (a == "--threaded-map-update") threaded = true;
        else if (a == "--camera" && i + 1 < argc) {
            for (const auto &c : Split(argv[++i], ',')) cam.push_back(atof(c.c_str()));
        } else {
            cerr << "Unknown argument: " << a << endl;
            WriteStatus(status_path, "config_error", "unknown argument " + a, 0, 0);
            return 2;
        }
    }
    if (cam.size() != 7 || !(cam[0] > 0) || !(cam[1] > 0) || !(cam[4] > 0) || !(cam[5] > 0) || !(cam[6] > 0)) {
        WriteStatus(status_path, "config_error", "--camera fx,fy,cx,cy,baseline,width,height required", 0, 0);
        return 2;
    }
    if (detection_rate < 0) {
        WriteStatus(status_path, "config_error", "--detection-rate must be >= 0", 0, 0);
        return 2;
    }

    rtabmap::ParametersMap params;
    string err;
    if (!LoadParams(params_path, params, err)) {
        cerr << err << endl;
        WriteStatus(status_path, "config_error", err, 0, 0);
        return 2;
    }
    const string db_path = output_dir + "/rtabmap.db";
    if (localize) {
        const auto it = params.find(rtabmap::Parameters::kMemIncrementalMemory());
        if (it == params.end() || (it->second != "false" && it->second != "0")) {
            WriteStatus(status_path, "config_error", "--localize needs Mem/IncrementalMemory=false", 0, 0);
            return 2;
        }
        if (!ifstream(db_path).good()) {
            WriteStatus(status_path, "config_error", "--localize needs a map database at " + db_path, 0, 0);
            return 2;
        }
    }
    vector<FrameEntry> frames;
    if (!LoadFrames(input_dir + "/frames.csv", frames) || frames.empty()) {
        WriteStatus(status_path, "input_error", "could not read stereo frames.csv", 0, 0);
        return 3;
    }

    ULogger::setType(ULogger::kTypeConsole);
    ULogger::setLevel(ULogger::kWarning);

    const Transform optical(0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0);
    const rtabmap::StereoCameraModel model("slambench", cam[0], cam[1], cam[2], cam[3], cam[4], optical,
                                           cv::Size(static_cast<int>(cam[5]), static_cast<int>(cam[6])));

    std::unique_ptr<rtabmap::Odometry> odom(rtabmap::Odometry::create(params));
    rtabmap::Rtabmap slam;
    if (!localize) std::remove(db_path.c_str());
    const auto init_start = chrono::steady_clock::now();
    slam.init(params, db_path);
    const double init_s = chrono::duration<double>(chrono::steady_clock::now() - init_start).count();
    size_t n_map_nodes = 0;
    if (localize) {
        const map<int, Transform> loaded = slam.getLocalOptimizedPoses();
        ofstream f(output_dir + "/map_loaded_nodes.csv");
        f << "node_id,tx,ty,tz,qx,qy,qz,qw\n" << fixed;
        for (const auto &kv : loaded) {
            if (kv.first <= 0) continue;  // landmarks have negative ids
            f << kv.first << ",";
            WritePose(f, kv.second * optical);
            f << "\n";
            ++n_map_nodes;
        }
        if (n_map_nodes == 0) {
            WriteStatus(status_path, "config_error", "the map database has no optimized nodes", 0, 0);
            return 2;
        }
        ofstream(output_dir + "/map_load.json") << "{\"init_time_s\": " << init_s << ", \"n_map_nodes\": "
                                                << n_map_nodes << ", \"wm_size\": " << slam.getWMSize() << "}\n";
    }

    ofstream log(output_dir + "/frame_log.csv");
    log << "frame_idx,timestamp_ns,odom_ok,features,inliers,matches,local_map_size,keyframe_added,"
           "slam_update,node_added,node_id,map_id,loop_closure_id,proximity_id,wm_size,"
           "odom_time_s,slam_time_s,track_time_s,coord_session,"
           "tx,ty,tz,qx,qy,qz,qw,odom_tx,odom_ty,odom_tz,odom_qx,odom_qy,odom_qz,odom_qw,"
           "localized,loc_event,loc_rejected,loc_correction_m\n";
    log << fixed;

    ofstream feat;
    if (log_features) {
        feat.open(output_dir + "/features.bin", ios::binary);
        feat.write("SBFEAT01", 8);
    }

    // Map-update side. In --threaded-map-update mode it runs on a worker thread (as the separate
    // odometry and rtabmap nodes of rtabmap_ros); `snap` is the map state the frame loop sees.
    struct UpdateResult {
        bool added = false, loc_event = false, loc_rejected = false;
        int loop_id = 0, prox_id = 0;
        double slam_s = 0.0, loc_corr = 0.0;
    };
    struct Snapshot {
        Transform correction;
        int last_node = 0, map_id = -1, wm_size = 0;
        long coord_session = 0;
        bool localized = false;
    };
    vector<NodeRecord> nodes;
    size_t n_updates = 0, n_loc_events = 0, n_dropped = 0;
    int root = 0;
    double map_thread_cpu_s = 0.0, odom_thread_cpu_s = 0.0;
    std::mutex mu;
    Snapshot snap;
    snap.correction = slam.getMapCorrection();
    snap.wm_size = slam.getWMSize();

    auto do_update = [&](rtabmap::SensorData &data, const Transform &odom_base, const cv::Mat &cov,
                         long long t_ns) {
        UpdateResult r;
        const double c0 = ThreadCpuSeconds();
        const auto s0 = chrono::steady_clock::now();
        slam.process(data, odom_base, cov);
        r.slam_s = chrono::duration<double>(chrono::steady_clock::now() - s0).count();
        map_thread_cpu_s += ThreadCpuSeconds() - c0;
        ++n_updates;
        Snapshot ns = snap;  // only this function writes snap, so reading it here needs no lock
        // Each update creates a signature, but RTAB-Map deletes it again when the robot moved
        // less than RGBD/LinearUpdate / AngularUpdate; only signatures still in memory are nodes.
        const int id = slam.getLastLocationId();
        if (!localize && id > 0 && id != ns.last_node && slam.getMemory() && slam.getMemory()->getSignature(id)) {
            r.added = true;
            ns.last_node = id;
            const int map_id = slam.getMemory() ? slam.getMemory()->getMapId(id) : -1;
            nodes.push_back({id, map_id, t_ns, odom_base * optical});
        }
        if (!localize) {
            const int gr = GraphRoot(slam.getLocalOptimizedPoses());
            if (gr != 0 && root != 0 && gr != root) ++ns.coord_session;
            if (gr != 0) root = gr;
        }
        const rtabmap::Statistics &su = slam.getStatistics();
        r.loop_id = su.loopClosureId();
        r.prox_id = su.proximityDetectionId();
        r.loc_event = r.loop_id > 0 || r.prox_id > 0;
        const auto &sd = su.data();
        const auto rj = sd.find(rtabmap::Statistics::kLoopRejectedHypothesis());
        r.loc_rejected = rj != sd.end() && rj->second > 0.0f;
        const auto cn = sd.find(rtabmap::Statistics::kLoopOdom_correction_norm());
        if (r.loc_event && cn != sd.end()) r.loc_corr = cn->second;
        if (r.loc_event) {
            ns.localized = true;
            ++n_loc_events;
        }
        ns.correction = slam.getMapCorrection();
        ns.map_id = (ns.last_node > 0 && slam.getMemory()) ? slam.getMemory()->getMapId(ns.last_node) : -1;
        ns.wm_size = slam.getWMSize();
        {
            std::lock_guard<std::mutex> lk(mu);
            snap = ns;
        }
        return r;
    };

    struct Request {
        size_t i = 0;
        long long t_ns = 0;
        rtabmap::SensorData data;
        Transform odom;
        cv::Mat cov;
    };
    std::condition_variable cv_req;
    std::unique_ptr<Request> pending;
    bool stop = false;
    map<size_t, UpdateResult> done;  // finished updates by frame index, until their row is written
    set<size_t> dropped;             // update requests replaced while the worker was busy
    std::thread worker;
    if (threaded) {
        worker = std::thread([&] {
            for (;;) {
                std::unique_ptr<Request> req;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv_req.wait(lk, [&] { return pending || stop; });
                    if (!pending) return;
                    req = std::move(pending);
                }
                const UpdateResult r = do_update(req->data, req->odom, req->cov, req->t_ns);
                std::lock_guard<std::mutex> lk(mu);
                done[req->i] = r;
            }
        });
    }
    auto finish_worker = [&] {
        if (!worker.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_req.notify_all();
        worker.join();  // the worker first finishes the pending request
    };
    struct Finally {
        std::function<void()> f;
        ~Finally() { f(); }
    } join_on_exit{finish_worker};

    struct Row {
        size_t i;
        long long t_ns;
        bool odom_ok, kf_added, requested;
        int features, inliers, matches, local_map_size;
        Snapshot s;
        double odom_s;
        Transform odom_base;
    };
    std::deque<Row> rows;
    auto write_row = [&](const Row &r, const UpdateResult *u) {
        log << r.i << "," << r.t_ns << "," << (r.odom_ok ? 1 : 0) << "," << r.features << ","
            << r.inliers << "," << r.matches << "," << r.local_map_size << ","
            << (r.kf_added ? 1 : 0) << "," << (u ? 1 : 0) << "," << (u && u->added ? 1 : 0) << ","
            << r.s.last_node << "," << r.s.map_id << "," << (u ? u->loop_id : 0) << ","
            << (u ? u->prox_id : 0) << "," << r.s.wm_size << ","
            << setprecision(6) << r.odom_s << "," << (u ? u->slam_s : 0.0) << ","
            << r.odom_s + (u && !threaded ? u->slam_s : 0.0) << "," << r.s.coord_session << ",";
        if (r.odom_ok) {
            WritePose(log, r.s.correction * r.odom_base * optical);
            log << ",";
            WritePose(log, r.odom_base * optical);
        } else {
            log << ",,,,,,,,,,,,,";
        }
        log << "," << (r.s.localized || (u && u->loc_event) ? 1 : 0) << "," << (u && u->loc_event ? 1 : 0)
            << "," << (u && u->loc_rejected ? 1 : 0) << "," << setprecision(6) << (u ? u->loc_corr : 0.0)
            << "\n";
    };
    // Rows are written in frame order once their map update (if any) has finished or was dropped.
    auto flush_rows = [&](bool all) {
        while (!rows.empty()) {
            const Row &r = rows.front();
            UpdateResult u;
            bool has_u = false;
            if (r.requested) {
                std::lock_guard<std::mutex> lk(mu);
                const auto it = done.find(r.i);
                if (it != done.end()) {
                    u = it->second;
                    has_u = true;
                    done.erase(it);
                } else if (!dropped.erase(r.i) && !all) {
                    break;
                }
            }
            write_row(r, has_u ? &u : nullptr);
            rows.pop_front();
        }
        log.flush();
    };
    auto fail = [&](const string &status, const string &msg, size_t fed) {
        finish_worker();
        flush_rows(true);
        WriteStatus(status_path, status, msg, frames.size(), fed, n_updates, nodes.size());
    };

    const auto wall_start = chrono::steady_clock::now();
    const double loop_cpu0 = ProcessCpuSeconds();
    size_t fed = 0;
    long long last_update_ns = 0;
    bool any_update = false;
    cv::Mat pending_cov;
    const long long update_period_ns = detection_rate > 0 ? static_cast<long long>(1e9 / detection_rate) : 0;

    for (size_t i = 0; i < frames.size(); ++i) {
        const auto &fr = frames[i];
        cv::Mat im_l = cv::imread(input_dir + "/" + fr.left, cv::IMREAD_UNCHANGED);
        cv::Mat im_r = cv::imread(input_dir + "/" + fr.right, cv::IMREAD_UNCHANGED);
        if (im_l.empty() || im_r.empty()) {
            cerr << "Failed to load image(s) for frame " << i << endl;
            fail("input_error", "failed to load image for frame " + to_string(i), fed);
            return 3;
        }
        if (im_l.channels() == 4) cv::cvtColor(im_l, im_l, cv::COLOR_BGRA2BGR);
        // RTAB-Map requires a grayscale right image; the left image may stay colour (used for the cloud).
        if (im_r.channels() == 4) cv::cvtColor(im_r, im_r, cv::COLOR_BGRA2GRAY);
        else if (im_r.channels() == 3) cv::cvtColor(im_r, im_r, cv::COLOR_BGR2GRAY);
        if (im_l.depth() != CV_8U || im_r.depth() != CV_8U) {
            fail("input_error", "frame " + to_string(i) + ": images must be 8-bit", fed);
            return 3;
        }

        const double stamp = fr.t_ns * 1e-9;
        rtabmap::SensorData data(im_l, im_r, model, static_cast<int>(i) + 1, stamp);
        rtabmap::OdometryInfo info;
        const double oc0 = ThreadCpuSeconds();
        const auto t0 = chrono::steady_clock::now();
        const Transform odom_base = odom->process(data, &info);
        const auto t1 = chrono::steady_clock::now();
        odom_thread_cpu_s += ThreadCpuSeconds() - oc0;
        const bool odom_ok = !odom_base.isNull();
        ++fed;

        // As RtabmapThread: keep the largest valid odometry covariance since the last map update
        // (independent of the odometry frame rate); 1.0 marks an unset covariance.
        if (odom_ok && !info.reg.covariance.empty()) {
            const double c00 = info.reg.covariance.at<double>(0, 0);
            if (std::isfinite(c00) && c00 != 1.0 && c00 > 0.0 &&
                (pending_cov.empty() || c00 > pending_cov.at<double>(0, 0)))
                pending_cov = info.reg.covariance.clone();
        }
        bool requested = false;
        if (odom_ok && (!any_update || update_period_ns == 0 || fr.t_ns - last_update_ns >= update_period_ns)) {
            cv::Mat cov = pending_cov.empty() ? cv::Mat::eye(6, 6, CV_64FC1) : pending_cov;
            pending_cov = cv::Mat();
            requested = true;
            any_update = true;
            last_update_ns = fr.t_ns;
            if (threaded) {
                // Latest wins (as RtabmapThread with a one-element buffer): a request still waiting
                // when the next one is due is dropped; its covariance is kept if it was larger.
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (pending) {
                        const double old_c = pending->cov.at<double>(0, 0), new_c = cov.at<double>(0, 0);
                        if (old_c != 1.0 && (new_c == 1.0 || old_c > new_c)) cov = pending->cov;
                        dropped.insert(pending->i);
                        ++n_dropped;
                    }
                    pending.reset(new Request{i, fr.t_ns, data, odom_base, cov});
                }
                cv_req.notify_one();
            } else {
                done[i] = do_update(data, odom_base, cov, fr.t_ns);
            }
        }
        Snapshot s;
        {
            std::lock_guard<std::mutex> lk(mu);
            s = snap;
        }
        rows.push_back({i, fr.t_ns, odom_ok, info.keyFrameAdded, requested, info.features, info.reg.inliers,
                        info.reg.matches, info.localMapSize, s, chrono::duration<double>(t1 - t0).count(),
                        odom_base});
        flush_rows(false);

        if (log_features) {
            const set<int> inl(info.reg.inliersIDs.begin(), info.reg.inliersIDs.end());
            const set<int> mat(info.reg.matchesIDs.begin(), info.reg.matchesIDs.end());
            const int64_t ts = fr.t_ns;
            const int32_t idx = static_cast<int32_t>(i), n = static_cast<int32_t>(info.words.size());
            feat.write(reinterpret_cast<const char *>(&ts), 8);
            feat.write(reinterpret_cast<const char *>(&idx), 4);
            feat.write(reinterpret_cast<const char *>(&n), 4);
            for (const auto &w : info.words) {
                const float xy[2] = {w.second.pt.x, w.second.pt.y};
                const uint8_t st = inl.count(w.first) ? 1 : (mat.count(w.first) ? 2 : 0);
                feat.write(reinterpret_cast<const char *>(xy), 8);
                feat.write(reinterpret_cast<const char *>(&st), 1);
            }
            if (i % 50 == 0) feat.flush();
        }

        if (playback_rate > 0.0 && i + 1 < frames.size()) {
            const double target = (frames[i + 1].t_ns - frames[0].t_ns) * 1e-9 / playback_rate;
            const double elapsed = chrono::duration<double>(chrono::steady_clock::now() - wall_start).count();
            if (target > elapsed) this_thread::sleep_for(chrono::duration<double>(target - elapsed));
        }
    }
    finish_worker();
    flush_rows(true);
    {
        const double loop_wall = chrono::duration<double>(chrono::steady_clock::now() - wall_start).count();
        ofstream f(output_dir + "/driver_timing.json");
        f << "{\n  \"threaded_map_update\": " << (threaded ? "true" : "false")
          << ",\n  \"n_update_requests\": " << n_updates + n_dropped
          << ",\n  \"n_update_requests_dropped\": " << n_dropped
          << ",\n  \"loop_wall_s\": " << loop_wall
          << ",\n  \"loop_process_cpu_s\": " << ProcessCpuSeconds() - loop_cpu0
          << ",\n  \"odometry_call_cpu_s\": " << odom_thread_cpu_s
          << ",\n  \"map_update_call_cpu_s\": " << map_thread_cpu_s
          << ",\n  \"note\": \"loop_process_cpu_s = user+system CPU of the whole driver process during the "
             "frame loop (all threads, excludes startup/map load and saving); the *_call_cpu_s are the "
             "CPU times of the calling thread inside odometry->process() and Rtabmap::process() only "
             "(library-internal worker threads are not included)\"\n}\n";
    }
    if (log_features) feat.close();
    if (localize) {
        // The map copy is never saved back: the run must not alter the map it is evaluated in.
        slam.close(false);
        WriteStatus(status_path, "completed", "localization events: " + to_string(n_loc_events),
                    frames.size(), fed, n_updates, 0);
        return 0;
    }

    // Final graph = the optimized poses of the current map (what Rtabmap::close() saves).
    const map<int, Transform> final_poses = slam.getLocalOptimizedPoses();
    {
        ofstream f(output_dir + "/nodes.csv");
        f << "node_id,map_id,timestamp_ns,in_final,odom_tx,odom_ty,odom_tz,odom_qx,odom_qy,odom_qz,odom_qw,"
             "tx,ty,tz,qx,qy,qz,qw\n" << fixed;
        for (const auto &n : nodes) {
            const auto it = final_poses.find(n.id);
            f << n.id << "," << n.map_id << "," << n.t_ns << "," << (it != final_poses.end() ? 1 : 0) << ",";
            WritePose(f, n.odom_cam);
            f << ",";
            WritePose(f, it != final_poses.end() ? it->second * optical : Transform());
            f << "\n";
        }
    }
    {
        map<int, Transform> poses;
        multimap<int, rtabmap::Link> links;
        slam.getGraph(poses, links, true, false);
        ofstream f(output_dir + "/links.csv");
        f << "from,to,type\n";
        for (const auto &kv : links)
            f << kv.second.from() << "," << kv.second.to() << "," << rtabmap::Link::typeName(kv.second.type()) << "\n";
    }
    slam.close(true);
    WriteStatus(status_path, "completed", "", frames.size(), fed, n_updates, nodes.size());
    return 0;
}
