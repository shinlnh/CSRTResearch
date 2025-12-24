#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../inc/csrt_tracker.hpp"

namespace fs = std::filesystem;

struct SequenceResult {
    std::string name;
    std::vector<float> overlaps;
    std::vector<float> distances;
    float auc = 0.0f;
    float precision_20px = 0.0f;
};

float ComputeIoU(const cv::Rect &a, const cv::Rect &b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }
    float inter = static_cast<float>((x2 - x1) * (y2 - y1));
    float union_area = static_cast<float>(a.width * a.height + b.width * b.height) - inter;
    return inter / union_area;
}

float ComputeCenterDistance(const cv::Rect &a, const cv::Rect &b) {
    cv::Point2f ca(a.x + a.width / 2.0f, a.y + a.height / 2.0f);
    cv::Point2f cb(b.x + b.width / 2.0f, b.y + b.height / 2.0f);
    return static_cast<float>(cv::norm(ca - cb));
}

bool LoadGroundTruth(const fs::path &anno_path, std::vector<cv::Rect> &gt_boxes) {
    std::ifstream file(anno_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << anno_path << "\n";
        return false;
    }

    gt_boxes.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int x = 0;
        int y = 0;
        int w = 0;
        int h = 0;
        char comma = 0;

        if (iss >> x >> comma >> y >> comma >> w >> comma >> h) {
            gt_boxes.emplace_back(x, y, w, h);
        } else {
            iss.clear();
            iss.str(line);
            if (iss >> x >> y >> w >> h) {
                gt_boxes.emplace_back(x, y, w, h);
            }
        }
    }

    return !gt_boxes.empty();
}

bool GetImageFiles(const fs::path &img_dir, std::vector<fs::path> &img_files) {
    img_files.clear();
    try {
        for (const auto &entry : fs::directory_iterator(img_dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    img_files.push_back(entry.path());
                }
            }
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
        return false;
    }

    std::sort(img_files.begin(), img_files.end());
    return !img_files.empty();
}

SequenceResult TrackSequence(const std::string &seq_name, const fs::path &seq_dir) {
    SequenceResult result;
    result.name = seq_name;

    std::cout << "  Processing: " << seq_name << "..." << std::flush;

    fs::path gt_path = seq_dir / "groundtruth_rect.txt";
    std::vector<cv::Rect> gt_boxes;
    if (!LoadGroundTruth(gt_path, gt_boxes)) {
        std::cerr << " [SKIP: No GT]\n";
        return result;
    }

    fs::path img_dir = seq_dir / "img";
    std::vector<fs::path> img_files;
    if (!GetImageFiles(img_dir, img_files)) {
        std::cerr << " [SKIP: No images]\n";
        return result;
    }

    size_t num_frames = std::min(gt_boxes.size(), img_files.size());
    cv::Mat first_frame = cv::imread(img_files[0].string());
    if (first_frame.empty()) {
        std::cerr << " [SKIP: Cannot read first frame]\n";
        return result;
    }

    csrt::CsrtTracker tracker;
    if (!tracker.Init(first_frame, gt_boxes[0])) {
        std::cerr << " [SKIP: Init failed]\n";
        return result;
    }

    for (size_t i = 1; i < num_frames; ++i) {
        cv::Mat frame = cv::imread(img_files[i].string());
        if (frame.empty()) {
            break;
        }

        cv::Rect pred_box;
        if (!tracker.Update(frame, pred_box)) {
            continue;
        }

        float iou = ComputeIoU(pred_box, gt_boxes[i]);
        float dist = ComputeCenterDistance(pred_box, gt_boxes[i]);
        result.overlaps.push_back(iou);
        result.distances.push_back(dist);
    }

    if (result.overlaps.empty()) {
        std::cerr << " [SKIP: No results]\n";
        return result;
    }

    const int num_thresholds = 50;
    std::vector<float> success_rates;
    for (int t = 0; t <= num_thresholds; ++t) {
        float threshold = t / static_cast<float>(num_thresholds);
        int count = static_cast<int>(std::count_if(result.overlaps.begin(), result.overlaps.end(),
            [threshold](float iou) { return iou > threshold; }));
        float rate = count / static_cast<float>(result.overlaps.size());
        success_rates.push_back(rate);
    }
    result.auc = std::accumulate(success_rates.begin(), success_rates.end(), 0.0f) / success_rates.size();

    int count_20px = static_cast<int>(std::count_if(result.distances.begin(), result.distances.end(),
        [](float d) { return d <= 20.0f; }));
    result.precision_20px = count_20px / static_cast<float>(result.distances.size());

    std::cout << " [AUC=" << std::fixed << std::setprecision(3) << result.auc
              << ", P@20=" << result.precision_20px << "]\n";
    return result;
}

int main(int argc, char **argv) {
    std::cout << "================================================================================\n";
    std::cout << "CSRT From Scratch - OTB-100 Evaluation\n";
    std::cout << "================================================================================\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --dataset /path/to/OTB100\n";
        return 1;
    }

    fs::path dataset_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dataset" && i + 1 < argc) {
            dataset_path = argv[i + 1];
            ++i;
        }
    }

    if (!fs::exists(dataset_path)) {
        std::cerr << "Dataset not found: " << dataset_path << "\n";
        return 1;
    }

    std::cout << "Dataset: " << dataset_path << "\n\n";

    std::vector<fs::path> sequences;
    for (const auto &entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_directory()) {
            sequences.push_back(entry.path());
        }
    }

    std::sort(sequences.begin(), sequences.end());
    std::cout << "Found " << sequences.size() << " sequences\n\n";

    std::vector<SequenceResult> results;
    for (const auto &seq_path : sequences) {
        auto result = TrackSequence(seq_path.filename().string(), seq_path);
        if (!result.overlaps.empty()) {
            results.push_back(result);
        }
    }

    if (results.empty()) {
        std::cerr << "No valid results!\n";
        return 1;
    }

    float mean_auc = 0.0f;
    float mean_precision = 0.0f;
    for (const auto &res : results) {
        mean_auc += res.auc;
        mean_precision += res.precision_20px;
    }

    mean_auc /= results.size();
    mean_precision /= results.size();

    std::cout << "\n================================================================================\n";
    std::cout << "OTB-100 Results\n";
    std::cout << "================================================================================\n";
    std::cout << "Sequences: " << results.size() << "\n";
    std::cout << "AUC:       " << std::fixed << std::setprecision(3) << mean_auc << "\n";
    std::cout << "P@20px:    " << std::fixed << std::setprecision(3) << mean_precision << "\n";
    std::cout << "================================================================================\n";

    return 0;
}
