// Applies the driver's image preprocessing to one image (used by slam-benching tests).
//   slambench_preprocess_check <in.png> <out.png> [--clahe]
// Exit codes: 0 ok, 2 usage, 3 unsupported/unreadable input, 4 write failure.
#include "slambench_preprocess.h"

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--clahe")) {
        std::cerr << "Usage: slambench_preprocess_check <in> <out> [--clahe]" << std::endl;
        return 2;
    }
    cv::Mat im = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    slambench::PreprocessStats stats;
    const std::string err = slambench::PreprocessForOrb(im, argc == 4, slambench::MakeClahe(), stats);
    if (!err.empty()) {
        std::cerr << "unsupported input: " << err << std::endl;
        return 3;
    }
    if (!cv::imwrite(argv[2], im)) return 4;
    std::cout << "channels=" << im.channels() << " color_to_gray=" << stats.n_color_to_gray
              << " clahe=" << stats.n_clahe << std::endl;
    return 0;
}
