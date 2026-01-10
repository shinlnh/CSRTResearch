#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../inc/csrt_tracker.hpp"

namespace {

std::vector<cv::Rect> LoadGroundTruth(const std::string &filename) {
    std::vector<cv::Rect> gt;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream iss(line);
        int x = 0;
        int y = 0;
        int w = 0;
        int h = 0;
        if (iss >> x >> y >> w >> h) {
            gt.emplace_back(x, y, w, h);
        }
    }
    return gt;
}

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

float ComputeCenterDist(const cv::Rect &a, const cv::Rect &b) {
    cv::Point2f ca(a.x + a.width / 2.0f, a.y + a.height / 2.0f);
    cv::Point2f cb(b.x + b.width / 2.0f, b.y + b.height / 2.0f);
    return static_cast<float>(cv::norm(ca - cb));
}

}  // namespace

int main() {
    std::string base_path = "E:/SourceCode/C2P/Project/CSRTResearch/otb100/OTB-dataset/OTB100/Basketball/Basketball";

    std::cout << "================================================================================\n";
    std::cout << "Testing CSRT From Scratch on Basketball (First 50 frames)\n";
    std::cout << "================================================================================\n";

    auto gt_boxes = LoadGroundTruth(base_path + "/groundtruth_rect.txt");
    std::cout << "\nLoaded " << gt_boxes.size() << " GT boxes\n";

    std::vector<std::string> img_files;
    for (int i = 1; i <= 50 && i <= static_cast<int>(gt_boxes.size()); ++i) {
        char filename[256];
        std::snprintf(filename, sizeof(filename), "%s/img/%04d.jpg", base_path.c_str(), i);
        img_files.push_back(filename);
    }

    csrt::CsrtTracker tracker;
    cv::Mat frame = cv::imread(img_files[0]);
    if (frame.empty()) {
        std::cerr << "Cannot read first frame!\n";
        return 1;
    }

    std::cout << "Frame size: " << frame.size() << "\n";
    std::cout << "Init bbox: " << gt_boxes[0] << "\n";

    if (!tracker.Init(frame, gt_boxes[0])) {
        std::cerr << "Initialization failed!\n";
        return 1;
    }

    std::cout << "\nTracking...\n";

    std::vector<float> ious;
    std::vector<float> dists;

    for (size_t i = 1; i < std::min(img_files.size(), gt_boxes.size()); ++i) {
        frame = cv::imread(img_files[i]);
        if (frame.empty()) {
            break;
        }

        cv::Rect pred_box;
        if (!tracker.Update(frame, pred_box)) {
            std::cout << "  Frame " << (i + 1) << " - Tracking failed!\n";
            continue;
        }

        float iou = ComputeIoU(pred_box, gt_boxes[i]);
        float dist = ComputeCenterDist(pred_box, gt_boxes[i]);
        ious.push_back(iou);
        dists.push_back(dist);

        if ((i + 1) % 10 == 0 || i == img_files.size() - 1) {
            std::cout << "  Frame " << (i + 1) << "/" << img_files.size()
                      << " - IoU: " << std::fixed << std::setprecision(3) << iou
                      << ", Dist: " << std::fixed << std::setprecision(1) << dist << "px\n";
        }
    }

    float mean_iou = 0.0f;
    float mean_dist = 0.0f;
    int precision_20 = 0;

    for (size_t i = 0; i < ious.size(); ++i) {
        mean_iou += ious[i];
        mean_dist += dists[i];
        if (dists[i] <= 20.0f) {
            precision_20++;
        }
    }

    mean_iou /= ious.size();
    mean_dist /= dists.size();
    float p20 = precision_20 / static_cast<float>(ious.size());

    int num_thresholds = 50;
    float auc = 0.0f;
    for (int t = 0; t <= num_thresholds; ++t) {
        float threshold = t / static_cast<float>(num_thresholds);
        int count = 0;
        for (float iou : ious) {
            if (iou > threshold) {
                count++;
            }
        }
        auc += count / static_cast<float>(ious.size());
    }
    auc /= (num_thresholds + 1);

    std::cout << "\n================================================================================\n";
    std::cout << "Results (50 frames)\n";
    std::cout << "================================================================================\n";
    std::cout << "AUC:             " << std::fixed << std::setprecision(3) << auc << "\n";
    std::cout << "Precision@20px:  " << std::fixed << std::setprecision(3) << p20 << "\n";
    std::cout << "Mean IoU:        " << std::fixed << std::setprecision(3) << mean_iou << "\n";
    std::cout << "Mean Distance:   " << std::fixed << std::setprecision(1) << mean_dist << "px\n";
    std::cout << "================================================================================\n";

    return 0;
}
