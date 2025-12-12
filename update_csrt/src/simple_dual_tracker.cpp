/**
 * @file simple_dual_tracker.cpp
 * @brief Simplified dual-branch tracker using OpenCV TrackerCSRT + Deep features
 */

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;

class SimpleDualTracker {
public:
    SimpleDualTracker() : alpha_(0.6f), initialized_(false) {
        // Create OpenCV CSRT tracker (pure implementation)
        csrt_tracker_ = TrackerCSRT::create();
        
        // Load VGG16 for deep features
        try {
            vgg_net_ = dnn::readNetFromONNX("../models/vgg16_conv4_3.onnx");
            vgg_net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
            vgg_net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
            
            corr_net_ = dnn::readNetFromONNX("../models/corr_project.onnx");
            corr_net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
            corr_net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
            
            gate_net_ = dnn::readNetFromONNX("../models/adaptive_gating.onnx");
            gate_net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
            gate_net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
            
            has_deep_ = true;
            std::cout << "Deep networks loaded successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load deep networks: " << e.what() << std::endl;
            has_deep_ = false;
        }
    }
    
    bool init(const Mat& frame, const Rect& bbox) {
        // Initialize pure CSRT tracker (init returns void in OpenCV 4.x)
        try {
            csrt_tracker_->init(frame, bbox);
        } catch (const std::exception& e) {
            std::cerr << "CSRT initialization failed: " << e.what() << std::endl;
            return false;
        }
        
        current_bbox_ = bbox;
        initialized_ = true;
        
        std::cout << "SimpleDualTracker initialized on bbox: " << bbox << std::endl;
        return true;
    }
    
    bool track(const Mat& frame, Rect& bbox) {
        if (!initialized_) {
            return false;
        }
        
        // Track with pure CSRT
        Rect csrt_bbox;
        if (!csrt_tracker_->update(frame, csrt_bbox)) {
            std::cerr << "CSRT tracking failed" << std::endl;
            return false;
        }
        
        // If no deep features, just return CSRT result
        if (!has_deep_) {
            bbox = csrt_bbox;
            current_bbox_ = bbox;
            return true;
        }
        
        // Extract deep features for confidence estimation
        Mat patch = extractPatch(frame, csrt_bbox, Size(127, 127));
        
        // VGG16 features
        Mat blob = dnn::blobFromImage(patch, 1.0/255.0, Size(127, 127), 
                                     Scalar(123.68, 116.779, 103.939), false, false);
        vgg_net_.setInput(blob);
        Mat vgg_features = vgg_net_.forward();
        
        // CorrProject: 512 → 31 channels
        corr_net_.setInput(vgg_features);
        Mat deep_features = corr_net_.forward(); // [1, 31, 15, 15]
        
        // Compute deep feature magnitude as confidence proxy
        Mat deep_flat = deep_features.reshape(1, 1);
        double deep_confidence = cv::norm(deep_flat, NORM_L2) / (31.0 * 15.0 * 15.0);
        
        // For baseline: just use CSRT bbox (deep features for future blending)
        bbox = csrt_bbox;
        
        current_bbox_ = bbox;
        
        return true;
    }
    
    Mat extractPatch(const Mat& frame, const Rect& bbox, const Size& size) {
        // Extract and resize patch from frame
        Rect safe_bbox = bbox & Rect(0, 0, frame.cols, frame.rows);
        if (safe_bbox.area() == 0) {
            return Mat::zeros(size, CV_8UC3);
        }
        
        Mat patch = frame(safe_bbox).clone();
        Mat resized;
        cv::resize(patch, resized, size);
        
        return resized;
    }
    
private:
    Ptr<TrackerCSRT> csrt_tracker_;
    dnn::Net vgg_net_;
    dnn::Net corr_net_;
    dnn::Net gate_net_;
    
    Rect current_bbox_;
    float alpha_;
    bool initialized_;
    bool has_deep_;
};

int main(int argc, char** argv) {
    std::string base_path = "E:/SourceCode/C2P/Project/CSRTResearch/otb100/OTB-dataset/OTB100/Basketball/Basketball";
    
    std::cout << "==========================================" << std::endl;
    std::cout << "Simple Dual Tracker - Basketball Test" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Load ground truth
    std::ifstream gt_file(base_path + "/groundtruth_rect.txt");
    std::vector<Rect> gt_boxes;
    std::string line;
    
    while (std::getline(gt_file, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream iss(line);
        
        int x, y, w, h;
        if (iss >> x >> y >> w >> h) {
            gt_boxes.emplace_back(x, y, w, h);
        }
    }
    
    std::cout << "Loaded " << gt_boxes.size() << " GT boxes" << std::endl;
    
    // Create tracker
    SimpleDualTracker tracker;
    
    // Load first frame
    char filename[256];
    sprintf(filename, "%s/img/0001.jpg", base_path.c_str());
    Mat frame = imread(filename);
    
    if (frame.empty()) {
        std::cerr << "Cannot read first frame!" << std::endl;
        return 1;
    }
    
    std::cout << "Frame size: " << frame.size() << std::endl;
    std::cout << "Init bbox: " << gt_boxes[0] << std::endl;
    
    // Initialize
    if (!tracker.init(frame, gt_boxes[0])) {
        std::cerr << "Initialization failed!" << std::endl;
        return 1;
    }
    
    // Track first 50 frames
    int num_frames = std::min(50, (int)gt_boxes.size());
    std::vector<float> ious, dists;
    
    for (int i = 1; i < num_frames; ++i) {
        sprintf(filename, "%s/img/%04d.jpg", base_path.c_str(), i + 1);
        frame = imread(filename);
        
        if (frame.empty()) break;
        
        Rect pred_bbox;
        if (!tracker.track(frame, pred_bbox)) {
            std::cout << "Frame " << (i + 1) << " - Tracking failed!" << std::endl;
            continue;
        }
        
        // Compute IoU
        int x1 = std::max(pred_bbox.x, gt_boxes[i].x);
        int y1 = std::max(pred_bbox.y, gt_boxes[i].y);
        int x2 = std::min(pred_bbox.x + pred_bbox.width, gt_boxes[i].x + gt_boxes[i].width);
        int y2 = std::min(pred_bbox.y + pred_bbox.height, gt_boxes[i].y + gt_boxes[i].height);
        
        float iou = 0.0f;
        if (x2 > x1 && y2 > y1) {
            float inter = (x2 - x1) * (y2 - y1);
            float union_area = pred_bbox.area() + gt_boxes[i].area() - inter;
            iou = inter / union_area;
        }
        
        // Compute center distance
        Point2f pred_center(pred_bbox.x + pred_bbox.width / 2.0f, 
                           pred_bbox.y + pred_bbox.height / 2.0f);
        Point2f gt_center(gt_boxes[i].x + gt_boxes[i].width / 2.0f,
                         gt_boxes[i].y + gt_boxes[i].height / 2.0f);
        float dist = norm(pred_center - gt_center);
        
        ious.push_back(iou);
        dists.push_back(dist);
        
        if ((i + 1) % 10 == 0 || i == num_frames - 1) {
            std::cout << "Frame " << (i + 1) << "/" << num_frames
                      << " - IoU: " << std::fixed << std::setprecision(3) << iou
                      << ", Dist: " << std::fixed << std::setprecision(1) << dist << "px"
                      << std::endl;
        }
    }
    
    // Compute metrics
    float mean_iou = 0.0f, mean_dist = 0.0f;
    int precision_20 = 0;
    
    for (size_t i = 0; i < ious.size(); ++i) {
        mean_iou += ious[i];
        mean_dist += dists[i];
        if (dists[i] <= 20.0f) precision_20++;
    }
    
    mean_iou /= ious.size();
    mean_dist /= dists.size();
    float p20 = precision_20 / static_cast<float>(ious.size());
    
    // AUC
    int num_thresholds = 50;
    float auc = 0.0f;
    for (int t = 0; t <= num_thresholds; ++t) {
        float threshold = t / static_cast<float>(num_thresholds);
        int count = 0;
        for (float iou : ious) {
            if (iou > threshold) count++;
        }
        auc += count / static_cast<float>(ious.size());
    }
    auc /= (num_thresholds + 1);
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Results (" << num_frames - 1 << " frames)" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "AUC:             " << std::fixed << std::setprecision(3) << auc << std::endl;
    std::cout << "Precision@20px:  " << std::fixed << std::setprecision(3) << p20 << std::endl;
    std::cout << "Mean IoU:        " << std::fixed << std::setprecision(3) << mean_iou << std::endl;
    std::cout << "Mean Distance:   " << std::fixed << std::setprecision(1) << mean_dist << "px" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
