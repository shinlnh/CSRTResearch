#include "../inc/AdaptiveGating.hpp"
#include <iostream>

namespace update_csrt {

AdaptiveGating::AdaptiveGating(const Config& config)
    : config_(config), initialized_(false), last_alpha_(config.alpha_default) {
    
    if (!config_.use_adaptive_alpha) {
        // Use fixed alpha, no network needed
        initialized_ = true;
        std::cout << "AdaptiveGating: Using fixed alpha = " << config_.alpha_default << std::endl;
        return;
    }
    
    try {
        // Load AdaptiveGating ONNX model
        std::cout << "Loading AdaptiveGating model from: " << config_.adaptive_gate_onnx_path << std::endl;
        net_ = cv::dnn::readNetFromONNX(config_.adaptive_gate_onnx_path);
        
        if (net_.empty()) {
            throw std::runtime_error("Failed to load AdaptiveGating ONNX model");
        }
        
        // Set backend
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        initialized_ = true;
        
        std::cout << "AdaptiveGating network initialized successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing AdaptiveGating: " << e.what() << std::endl;
        std::cerr << "Falling back to fixed alpha = " << config_.alpha_default << std::endl;
        initialized_ = true;  // Can still work with fixed alpha
    }
}

AdaptiveGating::~AdaptiveGating() {
    // Cleanup handled by OpenCV
}

cv::Mat AdaptiveGating::extractContextFeatures(const cv::Mat& response_map,
                                                const cv::Mat& template_feat,
                                                const cv::Mat& search_feat) {
    // Compute simple context statistics for now
    // In full implementation, this would be a learned feature extractor
    
    std::vector<float> context;
    
    // 1. Response map statistics
    double max_response, min_response;
    cv::minMaxLoc(response_map, &min_response, &max_response);
    context.push_back(static_cast<float>(max_response));
    context.push_back(static_cast<float>(min_response));
    
    // 2. Peak-to-Sidelobe Ratio (PSR)
    cv::Scalar mean, stddev;
    cv::meanStdDev(response_map, mean, stddev);
    float psr = (max_response - mean[0]) / (stddev[0] + 1e-8);
    context.push_back(psr);
    
    // 3. Feature similarity (cosine between template and search)
    cv::Mat temp_flat = template_feat.reshape(1, 1);  // Flatten
    cv::Mat search_flat = search_feat.reshape(1, 1);
    double similarity = temp_flat.dot(search_flat) / 
                       (cv::norm(temp_flat) * cv::norm(search_flat) + 1e-8);
    context.push_back(static_cast<float>(similarity));
    
    // Convert to Mat (1 x N)
    cv::Mat context_mat(1, context.size(), CV_32F);
    for (size_t i = 0; i < context.size(); ++i) {
        context_mat.at<float>(0, i) = context[i];
    }
    
    return context_mat;
}

float AdaptiveGating::clampAlpha(float alpha) {
    return std::max(config_.alpha_min, std::min(config_.alpha_max, alpha));
}

float AdaptiveGating::computeAlpha(const cv::Mat& response_map,
                                   const cv::Mat& template_feat,
                                   const cv::Mat& search_feat) {
    if (!config_.use_adaptive_alpha || net_.empty()) {
        // Use fixed alpha
        last_alpha_ = config_.alpha_default;
        return last_alpha_;
    }
    
    try {
        // Extract context features
        cv::Mat context = extractContextFeatures(response_map, template_feat, search_feat);
        
        // Forward pass through gating network
        net_.setInput(context);
        cv::Mat output = net_.forward();
        
        // Extract alpha value
        float alpha = output.at<float>(0, 0);
        
        // Clamp to valid range
        alpha = clampAlpha(alpha);
        
        last_alpha_ = alpha;
        return alpha;
        
    } catch (const std::exception& e) {
        std::cerr << "Error computing alpha: " << e.what() << std::endl;
        last_alpha_ = config_.alpha_default;
        return last_alpha_;
    }
}

cv::Mat AdaptiveGating::blendFilters(const cv::Mat& h_csrt, 
                                     const cv::Mat& h_deep, 
                                     float alpha) {
    if (h_csrt.size != h_deep.size || h_csrt.type() != h_deep.type()) {
        std::cerr << "Filter size/type mismatch in blending" << std::endl;
        return h_csrt.clone();
    }
    
    // h_final = α·h_csrt + (1-α)·h_deep
    cv::Mat h_final;
    cv::addWeighted(h_csrt, alpha, h_deep, 1.0 - alpha, 0.0, h_final);
    
    return h_final;
}

} // namespace update_csrt
