#include "../inc/DeepFeatureExtractor.hpp"
#include <iostream>

namespace update_csrt {

DeepFeatureExtractor::DeepFeatureExtractor(const Config& config)
    : config_(config), initialized_(false), feature_size_{512, 0, 0} {
    
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
        
        // Enable for dummy test (will be set to true after successful test)
        initialized_ = true;
        
        // Test with dummy input to get output size
        cv::Mat dummy = cv::Mat::zeros(config_.template_size, CV_8UC3);
        cv::Mat dummy_features = extract(dummy);
        
        if (dummy_features.empty() || dummy_features.dims < 3) {
            throw std::runtime_error("Failed to extract features from dummy input");
        }
        
        feature_size_.channels = dummy_features.size[0];  // Channels
        feature_size_.height = dummy_features.size[1];    // Height
        feature_size_.width = dummy_features.size[2];     // Width
        
        std::cout << "VGG16 initialized. Feature size: " 
                  << feature_size_.channels << "x" 
                  << feature_size_.height << "x" 
                  << feature_size_.width << std::endl;
                  
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
    
    // Forward pass through VGG16 ONNX model
    // ONNX has single output (no need to specify layer name)
    net_.setInput(blob);
    cv::Mat features = net_.forward();  // Get default output
    
    // Output shape: (1, C, H, W) -> Squeeze batch dimension
    std::vector<int> size_vec = {features.size[1], features.size[2], features.size[3]};
    features = features.reshape(1, size_vec);
    
    return features;
}

cv::Mat DeepFeatureExtractor::extractMasked(const cv::Mat& image, const cv::Mat& mask) {
    // ========== KEY ARCHITECTURE: MASK AT IMAGE LEVEL ==========
    // Apply mask to IMAGE before VGG16, not to features after!
    // Background pixels → mean color → VGG doesn't learn background
    
    cv::Mat masked_image = applyMaskToImage(image, mask);
    
    // Extract features from masked image (object-only region)
    return extract(masked_image);
}

cv::Mat DeepFeatureExtractor::applyMaskToImage(const cv::Mat& image, const cv::Mat& mask) {
    if (mask.empty() || image.empty()) {
        return image.clone();
    }
    
    // Convert mask to binary (0 or 1)
    cv::Mat binary_mask;
    if (mask.type() == CV_32FC1) {
        // Already float, threshold at 0.5
        cv::threshold(mask, binary_mask, 0.5, 1.0, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U, 255.0);
    } else {
        // Assume CV_8UC1, threshold at 127
        cv::threshold(mask, binary_mask, 127, 255, cv::THRESH_BINARY);
    }
    
    // Resize mask to match image size if needed
    if (binary_mask.size() != image.size()) {
        cv::resize(binary_mask, binary_mask, image.size(), 0, 0, cv::INTER_NEAREST);
    }
    
    // Create masked image: background = ImageNet mean color
    cv::Mat masked_image = image.clone();
    
    // Set background pixels (mask == 0) to mean color
    // mean_ = (103.939, 116.779, 123.68) in BGR
    for (int y = 0; y < image.rows; ++y) {
        const uchar* mask_row = binary_mask.ptr<uchar>(y);
        cv::Vec3b* img_row = masked_image.ptr<cv::Vec3b>(y);
        
        for (int x = 0; x < image.cols; ++x) {
            if (mask_row[x] == 0) {
                // Background pixel → set to ImageNet mean
                img_row[x][0] = static_cast<uchar>(mean_[0]); // B
                img_row[x][1] = static_cast<uchar>(mean_[1]); // G
                img_row[x][2] = static_cast<uchar>(mean_[2]); // R
            }
            // Object pixels (mask == 255) → keep original
        }
    }
    
    return masked_image;
}

} // namespace update_csrt
