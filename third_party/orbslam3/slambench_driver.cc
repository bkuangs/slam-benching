// Headless offline driver for ORB-SLAM3 used by slam-benching.
//
// Reads the normalized offline input format written by slam-benching dataset
// adapters, feeds it to ORB-SLAM3 without the viewer, and writes per-frame
// diagnostics (tracking state, active map id, per-frame tracking time) in
// addition to ORB-SLAM3's own final trajectory files.
//
// Usage:
//   orbslam3_driver <vocabulary> <settings.yaml> <mode> <input_dir> <output_dir>
//                   [--playback-rate R] [--clahe]
//
//   mode: stereo | stereo_inertial | mono | mono_inertial
//   --playback-rate R : R > 0 throttles input to R x the recorded frame rate
//                       (1.0 = real time); R = 0 feeds frames as fast as possible.
//   --clahe           : apply CLAHE(3.0, 8x8) to grayscale input (disabled by default).
//
// Input (input_dir):
//   frames.csv : timestamp_ns,left_path,right_path   (right_path empty for mono)
//   imu.csv    : timestamp_ns,wx,wy,wz,ax,ay,az      (rad/s, m/s^2; inertial modes only)
//
// Output (output_dir):
//   frame_log.csv         per-frame tracking diagnostics (flushed every frame)
//   trajectory_euroc.txt  System::SaveTrajectoryEuRoC (largest map, final optimized poses)
//   keyframes_euroc.txt   System::SaveKeyFrameTrajectoryEuRoC
//   driver_status.json    exit status written by the driver itself

#include <System.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
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

void WriteStatus(const string &path, const string &status, const string &message,
                 size_t n_frames_total, size_t n_frames_fed) {
    ofstream f(path);
    f << "{\n  \"status\": \"" << status << "\",\n  \"message\": \"" << message
      << "\",\n  \"n_frames_total\": " << n_frames_total
      << ",\n  \"n_frames_fed\": " << n_frames_fed << "\n}\n";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 6) {
        cerr << "Usage: orbslam3_driver <vocabulary> <settings.yaml> <mode> <input_dir> <output_dir>"
                " [--playback-rate R] [--clahe]" << endl;
        return 2;
    }
    const string vocab = argv[1], settings = argv[2], mode = argv[3];
    const string input_dir = argv[4], output_dir = argv[5];
    double playback_rate = 0.0;
    bool use_clahe = false;
    for (int i = 6; i < argc; ++i) {
        string a = argv[i];
        if (a == "--playback-rate" && i + 1 < argc) playback_rate = atof(argv[++i]);
        else if (a == "--clahe") use_clahe = true;
        else { cerr << "Unknown argument: " << a << endl; return 2; }
    }
    const string status_path = output_dir + "/driver_status.json";

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
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    ofstream log(output_dir + "/frame_log.csv");
    log << "frame_idx,timestamp_ns,state,map_id,map_changed,track_time_s,"
           "tx,ty,tz,qx,qy,qz,qw\n";
    log << fixed;

    size_t imu_idx = 0;
    if (inertial) {
        // Discard IMU samples at or before the first frame (ORB-SLAM3 example convention).
        while (imu_idx < imu.size() && imu_t[imu_idx] <= frames[0].t_ns) ++imu_idx;
    }

    const auto wall_start = chrono::steady_clock::now();
    size_t fed = 0;
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
        if (im_l.depth() != CV_8U || (stereo && im_r.depth() != CV_8U)) {
            // ORB extraction requires 8-bit input; refuse instead of silently finding no features.
            cerr << "Frame " << i << " is not 8-bit (depth " << im_l.depth() << ")" << endl;
            log.flush();
            WriteStatus(status_path, "input_error", "input images must be 8-bit; convert in the dataset adapter",
                        frames.size(), fed);
            SLAM.Shutdown();
            return 3;
        }
        if (use_clahe) {
            if (im_l.channels() == 1) clahe->apply(im_l, im_l);
            if (stereo && im_r.channels() == 1) clahe->apply(im_r, im_r);
        }

        vector<ORB_SLAM3::IMU::Point> imu_batch;
        if (inertial && i > 0) {
            while (imu_idx < imu.size() && imu_t[imu_idx] <= fr.t_ns) imu_batch.push_back(imu[imu_idx++]);
        }

        const double t = fr.t_ns * 1e-9;
        const auto t0 = chrono::steady_clock::now();
        Sophus::SE3f Tcw;
        if (stereo) Tcw = SLAM.TrackStereo(im_l, im_r, t, imu_batch);
        else Tcw = SLAM.TrackMonocular(im_l, t, imu_batch);
        const auto t1 = chrono::steady_clock::now();
        const double track_s = chrono::duration<double>(t1 - t0).count();
        ++fed;

        const int state = SLAM.GetTrackingState();
        const unsigned long map_id = SLAM.GetCurrentMapId();
        const bool map_changed = SLAM.MapChanged();
        log << i << "," << fr.t_ns << "," << state << "," << map_id << "," << (map_changed ? 1 : 0)
            << "," << setprecision(6) << track_s;
        // Online (not final-optimized) camera pose; only meaningful while tracking is OK.
        if (state == 2 || state == 5) {
            const Sophus::SE3f Twc = Tcw.inverse();
            const Eigen::Vector3f p = Twc.translation();
            const Eigen::Quaternionf q = Twc.unit_quaternion();
            log << setprecision(9) << "," << p.x() << "," << p.y() << "," << p.z() << ","
                << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
        } else {
            log << ",,,,,,,\n";
        }
        log.flush();

        if (playback_rate > 0.0 && i + 1 < frames.size()) {
            const double target = (frames[i + 1].t_ns - frames[0].t_ns) * 1e-9 / playback_rate;
            const double elapsed = chrono::duration<double>(chrono::steady_clock::now() - wall_start).count();
            if (target > elapsed) this_thread::sleep_for(chrono::duration<double>(target - elapsed));
        }
    }

    SLAM.Shutdown();
    SLAM.SaveTrajectoryEuRoC(output_dir + "/trajectory_euroc.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC(output_dir + "/keyframes_euroc.txt");
    WriteStatus(status_path, "completed", "", frames.size(), fed);
    return 0;
}
