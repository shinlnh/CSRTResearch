/**
 * @file otb_eval.cpp
 * @brief OTB-100 Evaluation Tool for Updated CSRT Tracker
 * 
 * Outputs:
 * - AUC (Area Under Curve) for success plot
 * - Precision @ 20px threshold
 * 
 * Usage:
 *   otb_eval --dataset /path/to/OTB100
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "../inc/UpdatedCSRTTracker.hpp"

namespace fs = std::filesystem;
using namespace update_csrt;

struct SequenceResult {
    std::string name;
    std::vector<float> overlaps;  // IoU for each frame
    std::vector<float> distances; // Center distance for each frame
    float auc;
    float precision_20px;
};

/**
 * @brief Compute IoU between two rectangles
 */
float computeIoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float union_area = a.width * a.height + b.width * b.height - intersection;
    
    return intersection / union_area;
}

/**
 * @brief Compute center distance
 */
float computeCenterDistance(const cv::Rect& a, const cv::Rect& b) {
    cv::Point2f ca(a.x + a.width / 2.0f, a.y + a.height / 2.0f);
    cv::Point2f cb(b.x + b.width / 2.0f, b.y + b.height / 2.0f);
    return cv::norm(ca - cb);
}

/**
 * @brief Load ground truth annotations
 */
bool loadGroundTruth(const fs::path& anno_path, std::vector<cv::Rect>& gt_boxes) {
    std::ifstream file(anno_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << anno_path << std::endl;
        return false;
    }
    
    gt_boxes.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int x, y, w, h;
        char comma;
        
        // Try format: x,y,w,h
        if (iss >> x >> comma >> y >> comma >> w >> comma >> h) {
            gt_boxes.emplace_back(x, y, w, h);
        }
        // Try format: x y w h
        else {
            iss.clear();
            iss.str(line);
            if (iss >> x >> y >> w >> h) {
                gt_boxes.emplace_back(x, y, w, h);
            }
        }
    }
    
    return !gt_boxes.empty();
}

/**
 * @brief Get image file list
 */
bool getImageFiles(const fs::path& img_dir, std::vector<fs::path>& img_files) {
    img_files.clear();
    
    try {
        for (const auto& entry : fs::directory_iterator(img_dir)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    img_files.push_back(entry.path());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return false;
    }
    
    std::sort(img_files.begin(), img_files.end());
    return !img_files.empty();
}

/**
 * @brief Track one sequence
 */
SequenceResult trackSequence(const std::string& seq_name, const fs::path& seq_dir) {
    SequenceResult result;
    result.name = seq_name;
    
    std::cout << "  Processing: " << seq_name << "..." << std::flush;
    
    // Load ground truth
    fs::path gt_path = seq_dir / "groundtruth_rect.txt";
    std::vector<cv::Rect> gt_boxes;
    if (!loadGroundTruth(gt_path, gt_boxes)) {
        std::cerr << " [SKIP: No GT]" << std::endl;
        return result;
    }
    
    // Get image files
    fs::path img_dir = seq_dir / "img";
    std::vector<fs::path> img_files;
    if (!getImageFiles(img_dir, img_files)) {
        std::cerr << " [SKIP: No images]" << std::endl;
        return result;
    }
    
    if (gt_boxes.size() != img_files.size()) {
        std::cerr << " [WARN: GT=" << gt_boxes.size() 
                  << " != Images=" << img_files.size() << "]" << std::flush;
    }
    
    size_t num_frames = std::min(gt_boxes.size(), img_files.size());
    
    // Initialize tracker with first frame
    cv::Mat first_frame = cv::imread(img_files[0].string());
    if (first_frame.empty()) {
        std::cerr << " [SKIP: Cannot read first frame]" << std::endl;
        return result;
    }
    
    Config config;
    config.visualize = false;  // No visualization in batch mode
    UpdatedCSRTTracker tracker(config);
    
    tracker.initialize(first_frame, gt_boxes[0]);
    
    // Track remaining frames
    for (size_t i = 1; i < num_frames; ++i) {
        cv::Mat frame = cv::imread(img_files[i].string());
        if (frame.empty()) break;
        
        cv::Rect pred_box;
        if (!tracker.track(frame, pred_box)) {
            // Tracking failed - use last known box
            continue;
        }
        
        // Compute metrics
        float iou = computeIoU(pred_box, gt_boxes[i]);
        float dist = computeCenterDistance(pred_box, gt_boxes[i]);
        
        result.overlaps.push_back(iou);
        result.distances.push_back(dist);
    }
    
    // Compute AUC (success plot integral from 0 to 1)
    const int num_thresholds = 50;
    std::vector<float> success_rates;
    for (int t = 0; t <= num_thresholds; ++t) {
        float threshold = t / static_cast<float>(num_thresholds);
        int count = std::count_if(result.overlaps.begin(), result.overlaps.end(),
                                 [threshold](float iou) { return iou > threshold; });
        float rate = count / static_cast<float>(result.overlaps.size());
        success_rates.push_back(rate);
    }
    result.auc = std::accumulate(success_rates.begin(), success_rates.end(), 0.0f) 
                / success_rates.size();
    
    // Compute precision @ 20px
    int count_20px = std::count_if(result.distances.begin(), result.distances.end(),
                                   [](float d) { return d <= 20.0f; });
    result.precision_20px = count_20px / static_cast<float>(result.distances.size());
    
    std::cout << " [AUC=" << std::fixed << std::setprecision(3) << result.auc 
              << ", P@20=" << result.precision_20px << "]" << std::endl;
    
    return result;
}

/**
 * @brief Main evaluation loop
 */
int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Updated CSRT Tracker - OTB-100 Evaluation" << std::endl;
    std::cout << "================================================================================" << std::endl;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --dataset /path/to/OTB100" << std::endl;
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
        std::cerr << "Dataset not found: " << dataset_path << std::endl;
        return 1;
    }
    
    std::cout << "Dataset: " << dataset_path << std::endl;
    std::cout << std::endl;
    
    // Get sequence directories
    std::vector<fs::path> sequences;
    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_directory()) {
            sequences.push_back(entry.path());
        }
    }
    
    std::sort(sequences.begin(), sequences.end());
    
    std::cout << "Found " << sequences.size() << " sequences" << std::endl;
    std::cout << std::endl;
    
    // Track all sequences
    std::vector<SequenceResult> results;
    for (const auto& seq_path : sequences) {
        auto result = trackSequence(seq_path.filename().string(), seq_path);
        if (!result.overlaps.empty()) {
            results.push_back(result);
        }
    }
    
    // Compute overall metrics
    if (results.empty()) {
        std::cerr << "No valid results!" << std::endl;
        return 1;
    }
    
    float mean_auc = 0.0f;
    float mean_precision_20px = 0.0f;
    for (const auto& res : results) {
        mean_auc += res.auc;
        mean_precision_20px += res.precision_20px;
    }
    mean_auc /= results.size();
    mean_precision_20px /= results.size();
    
    std::cout << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "OTB-100 Results" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Sequences: " << results.size() << std::endl;
    std::cout << "AUC:       " << std::fixed << std::setprecision(3) << mean_auc << std::endl;
    std::cout << "P@20px:    " << std::fixed << std::setprecision(3) << mean_precision_20px << std::endl;
    std::cout << "================================================================================" << std::endl;
    
    return 0;
}
