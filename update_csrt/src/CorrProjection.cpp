#include "../inc/CorrProjection.hpp"
#include <iostream>

namespace update_csrt {

CorrProjection::CorrProjection(const Config& config)
    : config_(config), initialized_(false) {
    
    try {
        // Load CorrProject ONNX model
        std::cout << "Loading CorrProject model from: " << config_.corr_project_onnx_path << std::endl;
        net_ = cv::dnn::readNetFromONNX(config_.corr_project_onnx_path);
        
        if (net_.empty()) {
            throw std::runtime_error("Failed to load CorrProject ONNX model");
        }
        
        // Set backend
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        initialized_ = true;
        
        std::cout << "CorrProject network initialized successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing CorrProjection: " << e.what() << std::endl;
        initialized_ = false;
    }
}

CorrProjection::~CorrProjection() {
    // Cleanup handled by OpenCV
}

cv::Mat CorrProjection::prepareInput(const cv::Mat& features) {
    // Input should be (C, H, W), need to add batch dimension -> (1, C, H, W)
    std::vector<int> shape = {1, features.size[0], features.size[1], features.size[2]};
    cv::Mat input = features.reshape(1, shape);
    return input;
}

cv::Mat CorrProjection::forward(const cv::Mat& deep_features) {
    if (!initialized_) {
        std::cerr << "CorrProjection not initialized" << std::endl;
        return cv::Mat();
    }
    
    if (deep_features.empty()) {
        std::cerr << "Empty input features" << std::endl;
        return cv::Mat();
    }
    
    // Prepare input
    cv::Mat input = prepareInput(deep_features);
    
    // Forward pass
    net_.setInput(input);
    cv::Mat output = net_.forward();
    
    // Remove batch dimension: (1, C_out, H, W) -> (C_out, H, W)
    std::vector<int> out_shape = {output.size[1], output.size[2], output.size[3]};
    output = output.reshape(1, out_shape);
    
    return output;
}

} // namespace update_csrt
