// Headless offline driver for SP_SLAM3 (ORB-SLAM3 fork with SuperPoint features) used by slam-benching.
//
// Port of third_party/orbslam3/slambench_driver.cc to SP_SLAM3's pre-v1.0 ORB-SLAM3 API
// (Track*() returns a cv::Mat Tcw, no Sophus). Same input format, same command line and the same
// output files, except place_recognition.csv (the ORB-SLAM3 place-recognition log patch is not
// ported). The slambench API additions (map frame info, outlier flags, sparse map export) are in
// third_party/spslam3/patches/0001-*.patch.
//
// Usage:
//   spslam3_driver <vocabulary> <settings.yaml> <mode> <input_dir> <output_dir>
//                  [--playback-rate R] [--clahe] [--clahe-clip C] [--clahe-tile N] [--log-features]
//
//   mode: stereo | stereo_inertial | mono | mono_inertial (slam-benching only runs stereo)
//   --playback-rate R : R > 0 throttles input to R x the recorded frame rate
//                       (1.0 = real time); R = 0 feeds frames as fast as possible.
//   --clahe           : apply CLAHE (disabled by default). Colour input (BGR/BGRA as
//                       loaded by cv::imread) is converted to grayscale first; other formats
//                       are rejected (see slambench_preprocess.h).
//   --clahe-clip C    : CLAHE clip limit (default 3.0; lower = gentler contrast enhancement).
//   --clahe-tile N    : CLAHE tile grid N x N (default 8).
//   --log-features    : write features.bin (keypoints of every frame and their map-point status).
//
// Input (input_dir):
//   frames.csv : timestamp_ns,left_path,right_path   (right_path empty for mono)
//   imu.csv    : timestamp_ns,wx,wy,wz,ax,ay,az      (rad/s, m/s^2; inertial modes only)
//
// Output (output_dir):
//   frame_log.csv         per-frame tracking diagnostics and online pose (flushed every frame).
//                         coord_session numbers the coordinate frame the online pose is expressed
//                         in (see SameCoordinateFrame); -1 = the frame changed while this frame
//                         was being tracked, so its pose's frame is ambiguous.
//                         map_kfs / map_points: size of the active map after the frame.
//   trajectory_euroc.txt  System::SaveTrajectoryEuRoC (largest map, final optimized poses)
//   keyframes_euroc.txt   System::SaveKeyFrameTrajectoryEuRoC
//   map_points.csv, keyframes.csv  sparse map of every atlas map at shutdown; rows with
//                         in_final_frame=1 are in trajectory_euroc.txt's world frame
//   features.bin          (--log-features) little-endian records, after the 8-byte magic "SBFEAT01":
//                         int64 timestamp_ns, int32 frame_idx, int32 n, then n x
//                         {float32 x, float32 y, uint8 status}; x,y = keypoint in the input image
//                         (pixels); status 0 = no map point, 1 = matched inlier, 2 = matched outlier
//   driver_status.json    exit status written by the driver itself

#include <System.h>

#include "slambench_preprocess.h"

#include <torch/cuda.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;

namespace {

struct FrameEntry {
    long long t_ns;
    string left;
    string right;
};

vector<string> SplitCsv(const string &line) {
    vector<string> out;
    string cell;
    stringstream ss(line);
    while (getline(ss, cell, ',')) out.push_back(cell);
    if (!line.empty() && line.back() == ',') out.push_back("");
    return out;
}

bool LoadFrames(const string &path, vector<FrameEntry> &frames) {
    ifstream f(path);
    if (!f.is_open()) return false;
    string line;
    getline(f, line);  // header
    while (getline(f, line)) {
        if (line.empty()) continue;
        auto c = SplitCsv(line);
        if (c.size() < 2) return false;
        frames.push_back({stoll(c[0]), c[1], c.size() > 2 ? c[2] : ""});
    }
    return true;
}

bool LoadImu(const string &path, vector<long long> &t_ns, vector<ORB_SLAM3::IMU::Point> &meas) {
    ifstream f(path);
    if (!f.is_open()) return false;
    string line;
    getline(f, line);  // header
    while (getline(f, line)) {
        if (line.empty()) continue;
        auto c = SplitCsv(line);
        if (c.size() < 7) return false;
        long long t = stoll(c[0]);
        t_ns.push_back(t);
        meas.emplace_back(stof(c[4]), stof(c[5]), stof(c[6]),
                          stof(c[1]), stof(c[2]), stof(c[3]), t * 1e-9);
    }
    return true;
}

slambench::PreprocessStats g_pre;

// Same world frame: same Map object and id, and the origin keyframe unchanged (loop closure and
// bundle adjustment keep it fixed; merges switch the Map object; in-place re-expression such as
// IMU initialisation moves it).
bool SameCoordinateFrame(const ORB_SLAM3::System::MapFrameInfo &a, const ORB_SLAM3::System::MapFrameInfo &b) {
    if (a.map != b.map || a.map_id != b.map_id || a.has_origin != b.has_origin) return false;
    if (!a.has_origin) return true;
    if (a.origin_kf_id != b.origin_kf_id) return false;
    const Eigen::Vector3f ta(a.t_w_origin[0], a.t_w_origin[1], a.t_w_origin[2]);
    const Eigen::Vector3f tb(b.t_w_origin[0], b.t_w_origin[1], b.t_w_origin[2]);
    const Eigen::Quaternionf qa(a.q_w_origin[3], a.q_w_origin[0], a.q_w_origin[1], a.q_w_origin[2]);
    const Eigen::Quaternionf qb(b.q_w_origin[3], b.q_w_origin[0], b.q_w_origin[1], b.q_w_origin[2]);
    const float dt = (ta - tb).norm();
    const float scale = std::max(1.0f, ta.norm());
    return dt <= 1e-4f * scale && qa.normalized().angularDistance(qb.normalized()) <= 1e-4f;
}

void WriteStatus(const string &path, const string &status, const string &message,
                 size_t n_frames_total, size_t n_frames_fed) {
    ofstream f(path);
    f << "{\n  \"status\": \"" << status << "\",\n  \"message\": \"" << message
      << "\",\n  \"n_frames_total\": " << n_frames_total
      << ",\n  \"n_frames_fed\": " << n_frames_fed
      << ",\n  \"n_images_color_to_gray\": " << g_pre.n_color_to_gray
      << ",\n  \"n_images_clahe\": " << g_pre.n_clahe
      << ",\n  \"cuda_available\": " << (torch::cuda::is_available() ? "true" : "false")
      << ",\n  \"cuda_device_count\": " << torch::cuda::device_count() << "\n}\n";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 6) {
        cerr << "Usage: spslam3_driver <vocabulary> <settings.yaml> <mode> <input_dir> <output_dir>"
                " [--playback-rate R] [--clahe] [--clahe-clip C] [--clahe-tile N] [--log-features]" << endl;
        return 2;
    }
    const string vocab = argv[1], settings = argv[2], mode = argv[3];
    const string input_dir = argv[4], output_dir = argv[5];
    double playback_rate = 0.0;
    bool use_clahe = false, log_features = false;
    double clahe_clip = 3.0;
    int clahe_tile = 8;
    for (int i = 6; i < argc; ++i) {
        string a = argv[i];
        if (a == "--playback-rate" && i + 1 < argc) playback_rate = atof(argv[++i]);
        else if (a == "--clahe") use_clahe = true;
        else if (a == "--clahe-clip" && i + 1 < argc) clahe_clip = atof(argv[++i]);
        else if (a == "--clahe-tile" && i + 1 < argc) clahe_tile = atoi(argv[++i]);
        else if (a == "--log-features") log_features = true;
        else { cerr << "Unknown argument: " << a << endl; return 2; }
    }
    const string status_path = output_dir + "/driver_status.json";
    if (!(clahe_clip > 0.0) || clahe_tile < 1) {
        cerr << "--clahe-clip must be > 0 and --clahe-tile >= 1" << endl;
        WriteStatus(status_path, "config_error", "invalid CLAHE parameters", 0, 0);
        return 2;
    }

    ORB_SLAM3::System::eSensor sensor;
    bool stereo = false, inertial = false;
    if (mode == "stereo") { sensor = ORB_SLAM3::System::STEREO; stereo = true; }
    else if (mode == "stereo_inertial") { sensor = ORB_SLAM3::System::IMU_STEREO; stereo = true; inertial = true; }
    else if (mode == "mono") { sensor = ORB_SLAM3::System::MONOCULAR; }
    else if (mode == "mono_inertial") { sensor = ORB_SLAM3::System::IMU_MONOCULAR; inertial = true; }
    else {
        cerr << "Unsupported mode: " << mode << endl;
        WriteStatus(status_path, "config_error", "unsupported mode " + mode, 0, 0);
        return 2;
    }

    vector<FrameEntry> frames;
    if (!LoadFrames(input_dir + "/frames.csv", frames) || frames.empty()) {
        WriteStatus(status_path, "input_error", "could not read frames.csv", 0, 0);
        return 3;
    }
    vector<long long> imu_t;
    vector<ORB_SLAM3::IMU::Point> imu;
    if (inertial && (!LoadImu(input_dir + "/imu.csv", imu_t, imu) || imu.empty())) {
        WriteStatus(status_path, "input_error", "could not read imu.csv", frames.size(), 0);
        return 3;
    }

    ORB_SLAM3::System SLAM(vocab, settings, sensor, false);
    cv::Ptr<cv::CLAHE> clahe = slambench::MakeClahe(clahe_clip, clahe_tile);

    ofstream log(output_dir + "/frame_log.csv");
    log << "frame_idx,timestamp_ns,state,map_id,map_changed,track_time_s,coord_session,"
           "map_kfs,map_points,tx,ty,tz,qx,qy,qz,qw\n";
    log << fixed;

    ofstream feat;
    if (log_features) {
        feat.open(output_dir + "/features.bin", ios::binary);
        feat.write("SBFEAT01", 8);
    }

    size_t imu_idx = 0;
    if (inertial) {
        // Discard IMU samples at or before the first frame (ORB-SLAM3 example convention).
        while (imu_idx < imu.size() && imu_t[imu_idx] <= frames[0].t_ns) ++imu_idx;
    }

    const auto wall_start = chrono::steady_clock::now();
    size_t fed = 0;
    long coord_session = 0;
    bool have_frame_info = false;
    ORB_SLAM3::System::MapFrameInfo last_frame_info;
    for (size_t i = 0; i < frames.size(); ++i) {
        const auto &fr = frames[i];
        cv::Mat im_l = cv::imread(input_dir + "/" + fr.left, cv::IMREAD_UNCHANGED);
        cv::Mat im_r;
        if (stereo) im_r = cv::imread(input_dir + "/" + fr.right, cv::IMREAD_UNCHANGED);
        if (im_l.empty() || (stereo && im_r.empty())) {
            cerr << "Failed to load image(s) for frame " << i << endl;
            log.flush();
            WriteStatus(status_path, "input_error", "failed to load image for frame " + to_string(i),
                        frames.size(), fed);
            SLAM.Shutdown();
            return 3;
        }
        // ORB extraction requires 8-bit 1/3/4-channel input; refuse anything else explicitly
        // instead of silently finding no features or skipping preprocessing.
        string err = slambench::PreprocessForOrb(im_l, use_clahe, clahe, g_pre);
        if (err.empty() && stereo) err = slambench::PreprocessForOrb(im_r, use_clahe, clahe, g_pre);
        if (!err.empty()) {
            cerr << "Frame " << i << ": " << err << endl;
            log.flush();
            WriteStatus(status_path, "input_error", "frame " + to_string(i) + ": " + err,
                        frames.size(), fed);
            SLAM.Shutdown();
            return 3;
        }

        vector<ORB_SLAM3::IMU::Point> imu_batch;
        if (inertial && i > 0) {
            while (imu_idx < imu.size() && imu_t[imu_idx] <= fr.t_ns) imu_batch.push_back(imu[imu_idx++]);
        }

        const double t = fr.t_ns * 1e-9;
        const ORB_SLAM3::System::MapFrameInfo info_before = SLAM.GetCurrentMapFrameInfo();
        const auto t0 = chrono::steady_clock::now();
        cv::Mat Tcw;
        if (stereo) Tcw = SLAM.TrackStereo(im_l, im_r, t, imu_batch);
        else Tcw = SLAM.TrackMonocular(im_l, t, imu_batch);
        const auto t1 = chrono::steady_clock::now();
        const double track_s = chrono::duration<double>(t1 - t0).count();
        ++fed;

        const int state = SLAM.GetTrackingState();
        const ORB_SLAM3::System::MapFrameInfo info = SLAM.GetCurrentMapFrameInfo();
        const unsigned long map_id = info.map_id;
        const bool map_changed = SLAM.MapChanged();
        if (have_frame_info && !SameCoordinateFrame(last_frame_info, info)) ++coord_session;
        have_frame_info = true;
        last_frame_info = info;
        const long frame_session = SameCoordinateFrame(info_before, info) ? coord_session : -1;
        log << i << "," << fr.t_ns << "," << state << "," << map_id << "," << (map_changed ? 1 : 0)
            << "," << setprecision(6) << track_s << "," << frame_session
            << "," << info.n_keyframes << "," << info.n_map_points;
        // Online (not final-optimized) camera pose; only meaningful while tracking is OK.
        if ((state == 2 || state == 5) && Tcw.rows == 4 && Tcw.cols == 4) {
            cv::Mat Tcw32;
            Tcw.convertTo(Tcw32, CV_32F);
            Eigen::Matrix3f Rcw;
            Eigen::Vector3f tcw;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) Rcw(r, c) = Tcw32.at<float>(r, c);
                tcw(r) = Tcw32.at<float>(r, 3);
            }
            const Eigen::Matrix3f Rwc = Rcw.transpose();
            const Eigen::Vector3f p = -Rwc * tcw;
            const Eigen::Quaternionf q = Eigen::Quaternionf(Rwc).normalized();
            log << setprecision(9) << "," << p.x() << "," << p.y() << "," << p.z() << ","
                << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
        } else {
            log << ",,,,,,,\n";
        }
        log.flush();

        if (log_features) {
            const vector<cv::KeyPoint> kps = SLAM.GetTrackedKeyPointsUn();
            const vector<ORB_SLAM3::MapPoint*> mps = SLAM.GetTrackedMapPoints();
            const vector<unsigned char> outl = SLAM.GetTrackedOutlierFlags();
            const int64_t ts = fr.t_ns;
            const int32_t idx = static_cast<int32_t>(i), n = static_cast<int32_t>(kps.size());
            feat.write(reinterpret_cast<const char *>(&ts), 8);
            feat.write(reinterpret_cast<const char *>(&idx), 4);
            feat.write(reinterpret_cast<const char *>(&n), 4);
            for (int32_t k = 0; k < n; ++k) {
                const float xy[2] = {kps[k].pt.x, kps[k].pt.y};
                uint8_t st = 0;
                if (k < static_cast<int32_t>(mps.size()) && mps[k] && !mps[k]->isBad())
                    st = (k < static_cast<int32_t>(outl.size()) && outl[k]) ? 2 : 1;
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

    SLAM.Shutdown();
    SLAM.SaveTrajectoryEuRoC(output_dir + "/trajectory_euroc.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC(output_dir + "/keyframes_euroc.txt");
    if (log_features) feat.close();
    if (!SLAM.SaveSparseMapSlambench(output_dir)) cerr << "could not write sparse map" << endl;
    WriteStatus(status_path, "completed", "", frames.size(), fed);
    return 0;
}
