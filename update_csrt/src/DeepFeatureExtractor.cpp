#include "../inc/DeepFeatureExtractor.hpp"
#include <iostream>

namespace update_csrt {

DeepFeatureExtractor::DeepFeatureExtractor(const Config& config)
    : config_(config), initialized_(false), feature_size_(512, 0, 0) {
    
    try {
        // Load VGG16 ONNX model using OpenCV DNN
        std::cout << "Loading VGG16 model from: " << config_.vgg16_onnx_path << std::endl;
        net_ = cv::dnn::readNetFromONNX(config_.vgg16_onnx_path);
        
        if (net_.empty()) {
            throw std::runtime_error("Failed to load VGG16 ONNX model");
        }
        
        // Set backend (CPU or CUDA)
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        // Test with dummy input to get output size
        cv::Mat dummy = cv::Mat::zeros(config_.template_size, CV_8UC3);
        cv::Mat dummy_features = extract(dummy);
        
        feature_size_.x = dummy_features.size[0];  // Channels
        feature_size_.y = dummy_features.size[1];  // Height
        feature_size_.z = dummy_features.size[2];  // Width
        
        initialized_ = true;
        
        std::cout << "VGG16 initialized. Feature size: " 
                  << feature_size_.x << "x" 
                  << feature_size_.y << "x" 
                  << feature_size_.z << std::endl;
                  
    } catch (const std::exception& e) {
        std::cerr << "Error initializing DeepFeatureExtractor: " << e.what() << std::endl;
        initialized_ = false;
    }
}

DeepFeatureExtractor::~DeepFeatureExtractor() {
    // OpenCV handles cleanup automatically
}

cv::Mat DeepFeatureExtractor::preprocessImage(const cv::Mat& image) {
    // Convert to float and subtract ImageNet mean
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    
    // Subtract mean (BGR order for VGG)
    float_img -= mean_;
    
    // Create blob (N, C, H, W) format
    cv::Mat blob = cv::dnn::blobFromImage(float_img, scale_, 
                                          float_img.size(), 
                                          cv::Scalar(), 
                                          false,  // swapRB = false (already BGR)
                                          false); // crop = false
    
    return blob;
}

cv::Mat DeepFeatureExtractor::extract(const cv::Mat& image) {
    if (!initialized_) {
        std::cerr << "DeepFeatureExtractor not initialized" << std::endl;
        return cv::Mat();
    }
    
    // Preprocess image
    cv::Mat blob = preprocessImage(image);
    
    // Forward pass
    net_.setInput(blob);
    cv::Mat features = net_.forward(config_.feature_layer);
    
    // Output shape: (1, C, H, W) -> Squeeze batch dimension
    std::vector<int> size_vec = {features.size[1], features.size[2], features.size[3]};
    features = features.reshape(1, size_vec);
    
    return features;
}

cv::Mat DeepFeatureExtractor::extractMasked(const cv::Mat& image, const cv::Mat& mask) {
    // Extract features
    cv::Mat features = extract(image);
    
    if (features.empty()) {
        return cv::Mat();
    }
    
    // Apply mask to features
    return applyMaskToFeatures(features, mask);
}

cv::Mat DeepFeatureExtractor::applyMaskToFeatures(const cv::Mat& features, const cv::Mat& mask) {
    if (mask.empty() || features.empty()) {
        return features;
    }
    
    // Resize mask to match feature spatial dimensions
    cv::Mat mask_resized;
    cv::resize(mask, mask_resized, 
               cv::Size(features.size[2], features.size[1]), 
               0, 0, cv::INTER_NEAREST);
    
    // Convert mask to float [0, 1]
    cv::Mat mask_float;
    mask_resized.convertTo(mask_float, CV_32F, 1.0 / 255.0);
    
    // Apply mask to each channel
    cv::Mat masked_features = features.clone();
    int C = features.size[0];
    int H = features.size[1];
    int W = features.size[2];
    
    for (int c = 0; c < C; ++c) {
        // Extract channel c
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = masked_features(ranges).clone();
        channel = channel.reshape(1, H);  // Reshape to 2D (H, W)
        
        // Multiply by mask
        channel = channel.mul(mask_float);
        
        // Put back
        channel.reshape(1, 1).copyTo(masked_features(ranges));
    }
    
    return masked_features;
}

} // namespace update_csrt
