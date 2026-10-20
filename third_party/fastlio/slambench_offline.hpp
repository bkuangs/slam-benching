// slam-benching offline input + logging for FAST-LIO (laserMapping.cpp).
//
// Included by patches/0001-slambench-offline-driver.patch right before main(), so it uses
// laserMapping.cpp's globals (callbacks, state, buffers). Enabled by the ROS parameter
// slambench/input_dir; without it FAST-LIO runs unchanged (ROS topics).
//
// Input (slambench offline format, written by the dataset adapter):
//   frames.csv  timestamp_ns,left_path,right_path   timestamp = scan end, left_path = lidar/<ns>.bin
//   lidar/*.bin magic "SBLIDAR1", int64 start_ns, uint32 n, uint32 record size (20), then n records
//               {f32 x, f32 y, f32 z, u32 offset_ns, u8 reflectivity, u8 tag, u8 line, u8 reserved}
//   imu.csv     timestamp_ns,wx,wy,wz,ax,ay,az (rad/s, m/s^2)
// Every scan becomes a livox_ros_driver/CustomMsg and every IMU row a sensor_msgs/Imu, passed to
// FAST-LIO's own livox_pcl_cbk / imu_cbk in data-time order (scan available at its end time, IMU at
// its stamp). slambench/playback_rate > 0 releases them in (scaled) real time and FAST-LIO processes
// what is available, as with a live sensor (scans queue if processing is slower; none are dropped);
// 0 feeds as fast as FAST-LIO consumes them.
//
// Output (slambench/output_dir):
//   frame_log.csv      one row per input frame: status, LiDAR pose (world <- LiDAR) right after the
//                      scan's update, IMU pose, extrinsic, velocity, biases, sizes and timing
//   fastlio_map.ply    voxel centroids (slambench/map_voxel m) of every processed scan's
//                      motion-compensated points (feats_undistort) placed at the scan's updated pose
//   driver_status.json, driver_timing.json
#pragma once
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>

namespace slambench {

using Clock = std::chrono::steady_clock;

struct ImuRow { int64_t t; double w[3], a[3]; };
struct FrameRow { int64_t t; std::string path; };

#pragma pack(push, 1)
struct ScanHeader { char magic[8]; int64_t start_ns; uint32_t n_points; uint32_t record_size; };
struct ScanPoint { float x, y, z; uint32_t offset_ns; uint8_t reflectivity, tag, line, reserved; };
#pragma pack(pop)
static_assert(sizeof(ScanPoint) == 20, "scan record layout");

// Open-addressing voxel hash: per voxel the sum of its points and their count (24 bytes per slot).
class VoxelMap {
public:
    explicit VoxelMap(double voxel) : inv_(1.0 / voxel) { resize(1 << 20); }
    void add(double x, double y, double z) {
        int64_t k = key(x, y, z);
        size_t i = slot(k);
        if (slots_[i].k == kEmpty) {
            if ((n_ + 1) * 10 > slots_.size() * 6) { resize(slots_.size() * 2); i = slot(k); }
            slots_[i].k = k; ++n_;
        }
        slots_[i].s[0] += float(x); slots_[i].s[1] += float(y); slots_[i].s[2] += float(z); ++slots_[i].n;
    }
    size_t size() const { return n_; }
    size_t bytes() const { return slots_.size() * sizeof(Slot); }
    bool write_ply(const std::string &path, uint32_t min_count) const {
        size_t n = 0;
        for (const auto &s : slots_) n += (s.k != kEmpty && s.n >= min_count);
        FILE *f = fopen(path.c_str(), "wb");
        if (!f) return false;
        fprintf(f, "ply\nformat binary_little_endian 1.0\ncomment slam-benching FAST-LIO map: voxel "
                   "centroids, count = points merged\nelement vertex %zu\nproperty float x\n"
                   "property float y\nproperty float z\nproperty uint count\nend_header\n", n);
        for (const auto &s : slots_) {
            if (s.k == kEmpty || s.n < min_count) continue;
            float v[3] = {s.s[0] / s.n, s.s[1] / s.n, s.s[2] / s.n};
            fwrite(v, sizeof(float), 3, f);
            fwrite(&s.n, sizeof(uint32_t), 1, f);
        }
        return fclose(f) == 0;
    }
private:
    struct Slot { int64_t k; float s[3]; uint32_t n; };
    static constexpr int64_t kEmpty = INT64_MIN;
    int64_t key(double x, double y, double z) const {
        auto c = [&](double v) { return (int64_t(std::floor(v * inv_)) + (1 << 20)) & 0x1FFFFF; };
        return (c(x) << 42) | (c(y) << 21) | c(z);
    }
    size_t slot(int64_t k) const {
        uint64_t h = uint64_t(k) * 0x9E3779B97F4A7C15ull;
        size_t mask = slots_.size() - 1, i = size_t(h >> 20) & mask;
        while (slots_[i].k != kEmpty && slots_[i].k != k) i = (i + 1) & mask;
        return i;
    }
    void resize(size_t cap) {
        std::vector<Slot> old;
        old.swap(slots_);
        slots_.assign(cap, Slot{kEmpty, {0, 0, 0}, 0});
        for (const auto &s : old)
            if (s.k != kEmpty) slots_[slot(s.k)] = s;
    }
    double inv_;
    std::vector<Slot> slots_;
    size_t n_ = 0;
};

class Offline {
public:
    bool enabled = false;

    void init(ros::NodeHandle &nh) {
        nh.param<std::string>("slambench/input_dir", in_dir_, "");
        if (in_dir_.empty()) return;
        enabled = true;
        nh.param<std::string>("slambench/output_dir", out_dir_, ".");
        nh.param<double>("slambench/playback_rate", rate_, 1.0);
        nh.param<double>("slambench/map_voxel", map_voxel_, 0.02);
        nh.param<int>("slambench/map_min_count", map_min_count_, 1);
        if (map_voxel_ > 0) map_.reset(new VoxelMap(map_voxel_));
        read_frames();
        read_imu();
        log_.open(out_dir_ + "/frame_log.csv");
        log_ << "frame_idx,timestamp_ns,status,pose_valid,lidar_beg_time,lidar_end_time,n_imu,"
                "tx,ty,tz,qx,qy,qz,qw,imu_tx,imu_ty,imu_tz,imu_qx,imu_qy,imu_qz,imu_qw,"
                "ext_tx,ext_ty,ext_tz,ext_qx,ext_qy,ext_qz,ext_qw,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,"
                "n_points_raw,n_points_undistort,n_points_down,n_effective,"
                "track_time_s,preprocess_time_s,read_time_s,map_time_s,feed_wall_s,done_wall_s\n";
        log_.precision(9);
        status("running", "");
        wall0_ = Clock::now();
        if (!frames_.empty() || !imu_.empty())
            data0_ = std::min(frames_.empty() ? INT64_MAX : frames_.front().t, imu_.empty() ? INT64_MAX : imu_.front().t);
        ROS_INFO("slambench offline: %zu scans, %zu IMU samples from %s, playback rate %.2f",
                 frames_.size(), imu_.size(), in_dir_.c_str(), rate_);
    }

    // Stands in for ros::spinOnce(): hands the next due messages to FAST-LIO's callbacks.
    // Returns false once all input was given and the previous sync_packages() call failed.
    bool spin() {
        bool synced = synced_;
        synced_ = false;
        if (rate_ <= 0) {
            if (synced) return true;               // buffers may hold another complete package
            if (exhausted()) return false;
            feed_next();
            return true;
        }
        int64_t now = data_now();
        bool fed = false;
        while (!exhausted() && next_avail() <= now) { feed_next(); fed = true; }
        if (synced || fed) return true;
        if (exhausted()) return false;
        auto due = wall0_ + std::chrono::nanoseconds(int64_t((next_avail() - data0_) / rate_));
        std::this_thread::sleep_until(std::min(due, Clock::now() + std::chrono::milliseconds(5)));
        return true;
    }

    // After a successful sync_packages(): the package's scan is the oldest fed, unprocessed one.
    void begin_scan(size_t lidar_buffer_left, size_t n_imu) {
        synced_ = true;
        if (pending_.empty()) fail("sync_packages returned a scan that was never fed");
        cur_ = pending_.front();
        pending_.erase(pending_.begin());
        if (pending_.size() != lidar_buffer_left)
            fail("FAST-LIO dropped LiDAR scans from its buffer (loop back?)");
        cur_n_imu_ = n_imu;
        cur_status_ = "unknown";
        t_begin_ = Clock::now();
        active_ = true;
    }
    void set_status(const char *s) { cur_status_ = s; }

    // End of the package's processing (called for every outcome).
    void end_scan(bool ok) {
        if (!active_) return;
        active_ = false;
        auto t_end = Clock::now();
        double track = secs(t_end - t_begin_) + cur_.preprocess_s;
        double map_s = 0;
        if (ok) {
            ++n_ok_;
            if (map_) {
                auto t0 = Clock::now();
                PointType w;
                for (const auto &p : feats_undistort->points) {
                    pointBodyToWorld(&p, &w);
                    map_->add(w.x, w.y, w.z);
                }
                map_s = secs(Clock::now() - t0);
                map_time_ += map_s;
            }
        }
        if (!ok && n_ok_ == 0 && std::string(cur_status_) == "no_points") cur_status_ = "initializing";
        track_time_ += track;
        write_row(cur_, ok, track, map_s, secs(Clock::now() - wall0_));
    }

    void finish(const char *why) {
        end_scan(false);
        for (const auto &p : pending_) { cur_ = p; cur_status_ = "not_processed"; write_row(cur_, false, 0, 0, 0, true); }
        for (; next_frame_ < frames_.size(); ++next_frame_) {
            cur_ = Pending{next_frame_, frames_[next_frame_].t, 0, 0, 0, 0};
            cur_status_ = "not_fed";
            write_row(cur_, false, 0, 0, 0, true);
        }
        log_.close();
        bool map_ok = true;
        size_t map_voxels = 0, map_bytes = 0;
        double t_write = 0;
        if (map_) {
            auto t0 = Clock::now();
            map_ok = map_->write_ply(out_dir_ + "/fastlio_map.ply", uint32_t(std::max(1, map_min_count_)));
            t_write = secs(Clock::now() - t0);
            map_voxels = map_->size();
            map_bytes = map_->bytes();
        }
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        std::ofstream t(out_dir_ + "/driver_timing.json");
        t.precision(9);
        t << "{\"wall_s\": " << secs(Clock::now() - wall0_) << ", \"n_frames\": " << frames_.size()
          << ", \"n_fed\": " << next_frame_ << ", \"n_ok\": " << n_ok_
          << ", \"track_time_total_s\": " << track_time_ << ", \"read_time_total_s\": " << read_time_
          << ", \"map_accumulation_time_total_s\": " << map_time_ << ", \"map_write_s\": " << t_write
          << ", \"map_voxel_m\": " << map_voxel_ << ", \"map_voxels\": " << map_voxels
          << ", \"map_min_count\": " << map_min_count_ << ", \"map_hash_bytes\": " << map_bytes
          << ", \"playback_rate\": " << rate_ << ", \"omp_threads\": " << MP_PROC_NUM
          << ", \"self_maxrss_mb\": " << ru.ru_maxrss / 1024.0
          << ", \"self_cpu_user_s\": " << ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6
          << ", \"self_cpu_system_s\": " << ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6 << "}\n";
        if (!map_ok) status("error", "failed to write fastlio_map.ply");
        else status("completed", why);
    }

    [[noreturn]] void fail(const std::string &msg) {
        status("error", msg);
        ROS_FATAL("slambench offline: %s", msg.c_str());
        std::exit(3);
    }

private:
    struct Pending { size_t idx; int64_t t; double preprocess_s, read_s, feed_wall_s; size_t n_raw; };

    static double secs(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
    bool exhausted() const { return next_frame_ >= frames_.size() && next_imu_ >= imu_.size(); }
    int64_t next_avail() const {
        int64_t a = next_frame_ < frames_.size() ? frames_[next_frame_].t : INT64_MAX;
        int64_t b = next_imu_ < imu_.size() ? imu_[next_imu_].t : INT64_MAX;
        return std::min(a, b);
    }
    int64_t data_now() const { return data0_ + int64_t(secs(Clock::now() - wall0_) * 1e9 * rate_); }

    void feed_next() {
        bool imu_first = next_imu_ < imu_.size() &&
                         (next_frame_ >= frames_.size() || imu_[next_imu_].t <= frames_[next_frame_].t);
        if (imu_first) {
            const ImuRow &r = imu_[next_imu_++];
            sensor_msgs::Imu::Ptr m(new sensor_msgs::Imu());
            m->header.stamp.fromNSec(uint64_t(r.t));
            m->header.frame_id = "imu";
            m->angular_velocity.x = r.w[0]; m->angular_velocity.y = r.w[1]; m->angular_velocity.z = r.w[2];
            m->linear_acceleration.x = r.a[0]; m->linear_acceleration.y = r.a[1]; m->linear_acceleration.z = r.a[2];
            m->orientation_covariance[0] = -1;
            imu_cbk(m);
            return;
        }
        const FrameRow &fr = frames_[next_frame_];
        auto t0 = Clock::now();
        std::ifstream f(in_dir_ + "/" + fr.path, std::ios::binary);
        ScanHeader h{};
        if (!f.read(reinterpret_cast<char *>(&h), sizeof(h)) || std::string(h.magic, 8) != "SBLIDAR1" ||
            h.record_size != sizeof(ScanPoint))
            fail("bad scan file " + fr.path);
        std::vector<ScanPoint> pts(h.n_points);
        if (h.n_points && !f.read(reinterpret_cast<char *>(pts.data()), sizeof(ScanPoint) * h.n_points))
            fail("truncated scan file " + fr.path);
        livox_ros_driver::CustomMsg::Ptr m(new livox_ros_driver::CustomMsg());
        m->header.stamp.fromNSec(uint64_t(h.start_ns));
        m->header.frame_id = "livox_frame";
        m->timebase = uint64_t(h.start_ns);
        m->point_num = h.n_points;
        m->points.resize(h.n_points);
        for (uint32_t i = 0; i < h.n_points; ++i) {
            auto &d = m->points[i];
            const auto &s = pts[i];
            d.offset_time = s.offset_ns; d.x = s.x; d.y = s.y; d.z = s.z;
            d.reflectivity = s.reflectivity; d.tag = s.tag; d.line = s.line;
        }
        double read_s = secs(Clock::now() - t0);
        read_time_ += read_s;
        size_t before = lidar_buffer.size();
        auto t1 = Clock::now();
        livox_pcl_cbk(m);
        double pre_s = secs(Clock::now() - t1);
        if (lidar_buffer.size() != before + 1) fail("FAST-LIO did not buffer scan " + fr.path);
        pending_.push_back(Pending{next_frame_, fr.t, pre_s, read_s, secs(Clock::now() - wall0_), h.n_points});
        ++next_frame_;
    }

    void write_row(const Pending &p, bool ok, double track, double map_s, double done_wall, bool skipped = false) {
        auto &o = log_;
        o << p.idx << ',' << p.t << ',' << cur_status_ << ',' << int(ok) << ',';
        if (skipped) {
            o << ",,";
        } else {
            o << std::fixed << Measures.lidar_beg_time << ',' << Measures.lidar_end_time << ',';
            o.unsetf(std::ios::floatfield);
        }
        o << (skipped ? 0 : cur_n_imu_) << ',';
        if (ok) {
            Eigen::Matrix3d R = state_point.rot.toRotationMatrix();
            Eigen::Matrix3d Rl = state_point.offset_R_L_I.toRotationMatrix();
            Eigen::Vector3d t_l = R * state_point.offset_T_L_I + state_point.pos;
            Eigen::Quaterniond q_l(R * Rl), q_i(R), q_e(Rl);
            q_l.normalize(); q_i.normalize(); q_e.normalize();
            auto pose = [&](const Eigen::Vector3d &t, const Eigen::Quaterniond &q) {
                o << t.x() << ',' << t.y() << ',' << t.z() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << ',';
            };
            pose(t_l, q_l);
            pose(state_point.pos, q_i);
            pose(state_point.offset_T_L_I, q_e);
            const auto &v = state_point.vel;
            const auto &bg = state_point.bg;
            const auto &ba = state_point.ba;
            o << v(0) << ',' << v(1) << ',' << v(2) << ',' << bg(0) << ',' << bg(1) << ',' << bg(2) << ','
              << ba(0) << ',' << ba(1) << ',' << ba(2) << ',';
        } else {
            for (int i = 0; i < 30; ++i) o << ',';
        }
        if (skipped) o << p.n_raw << ",,,,";
        else o << p.n_raw << ',' << feats_undistort->size() << ',' << feats_down_size << ',' << (ok ? effct_feat_num : 0) << ',';
        o << (skipped ? 0.0 : track) << ',' << p.preprocess_s << ',' << p.read_s << ',' << map_s << ','
          << p.feed_wall_s << ',' << done_wall << '\n';
    }

    void status(const std::string &s, const std::string &msg) {
        std::ofstream f(out_dir_ + "/driver_status.json");
        std::string m;
        for (char c : msg) { if (c == '"' || c == '\\') m += '\\'; m += (c == '\n' ? ' ' : c); }
        f << "{\"status\": \"" << s << "\", \"message\": \"" << m << "\"}\n";
    }

    void read_frames() {
        std::ifstream f(in_dir_ + "/frames.csv");
        if (!f) fail("cannot open frames.csv in " + in_dir_);
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string ts, path;
            std::getline(ss, ts, ',');
            std::getline(ss, path, ',');
            frames_.push_back(FrameRow{std::stoll(ts), path});
        }
    }
    void read_imu() {
        std::ifstream f(in_dir_ + "/imu.csv");
        if (!f) fail("cannot open imu.csv in " + in_dir_ + " (FAST-LIO needs an IMU)");
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            ImuRow r{};
            if (sscanf(line.c_str(), "%ld,%lf,%lf,%lf,%lf,%lf,%lf", &r.t, &r.w[0], &r.w[1], &r.w[2],
                       &r.a[0], &r.a[1], &r.a[2]) != 7)
                fail("malformed imu.csv line: " + line);
            imu_.push_back(r);
        }
    }

    std::string in_dir_, out_dir_;
    double rate_ = 1.0, map_voxel_ = 0.02;
    int map_min_count_ = 1;
    std::unique_ptr<VoxelMap> map_;
    std::vector<FrameRow> frames_;
    std::vector<ImuRow> imu_;
    size_t next_frame_ = 0, next_imu_ = 0;
    std::vector<Pending> pending_;
    Pending cur_{};
    size_t cur_n_imu_ = 0, n_ok_ = 0;
    const char *cur_status_ = "unknown";
    bool synced_ = false, active_ = false;
    Clock::time_point wall0_, t_begin_;
    int64_t data0_ = 0;
    double track_time_ = 0, read_time_ = 0, map_time_ = 0;
    std::ofstream log_;
};

// Calls end_scan() when a package's processing ends, including FAST-LIO's early `continue`s.
struct ScanScope {
    Offline &o;
    bool ok = false;
    explicit ScanScope(Offline &off) : o(off) {}
    ~ScanScope() { if (o.enabled) o.end_scan(ok); }
};

}  // namespace slambench
