// Image preprocessing shared by the slam-benching ORB-SLAM3 driver and its check tool.
//
// Images are loaded with cv::imread(IMREAD_UNCHANGED), so colour images are in OpenCV's
// BGR / BGRA channel order. Without CLAHE the image is passed through unchanged and
// ORB-SLAM3 converts it to grayscale itself (settings use Camera.RGB: 0 = BGR).
// With CLAHE, colour images are first converted to grayscale with the same BGR
// convention, because CLAHE is defined on a single intensity channel.
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>

namespace slambench {

struct PreprocessStats {
    long long n_color_to_gray = 0;   // colour images converted before CLAHE
    long long n_clahe = 0;           // images CLAHE was applied to
};

// Returns an empty string on success, otherwise a reason the image cannot be used.
inline std::string PreprocessForOrb(cv::Mat &im, bool use_clahe, const cv::Ptr<cv::CLAHE> &clahe,
                                    PreprocessStats &stats) {
    if (im.empty()) return "empty image";
    if (im.depth() != CV_8U) return "image is not 8-bit (depth " + std::to_string(im.depth()) + ")";
    const int ch = im.channels();
    if (ch != 1 && ch != 3 && ch != 4)
        return "unsupported number of channels: " + std::to_string(ch) + " (expected 1, 3 or 4)";
    if (!use_clahe) return "";
    if (ch == 3) {
        cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
        ++stats.n_color_to_gray;
    } else if (ch == 4) {
        cv::cvtColor(im, im, cv::COLOR_BGRA2GRAY);
        ++stats.n_color_to_gray;
    }
    clahe->apply(im, im);
    ++stats.n_clahe;
    return "";
}

inline cv::Ptr<cv::CLAHE> MakeClahe() { return cv::createCLAHE(3.0, cv::Size(8, 8)); }

}  // namespace slambench
