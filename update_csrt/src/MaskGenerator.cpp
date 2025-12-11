#include "../inc/MaskGenerator.hpp"
#include <iostream>

namespace update_csrt {

MaskGenerator::MaskGenerator(const Config& config)
    : config_(config) {
    
    // Create morphological structuring element
    morph_kernel_ = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(config_.mask_morph_size, config_.mask_morph_size)
    );
    
    std::cout << "MaskGenerator initialized with threshold=" 
              << config_.mask_threshold << std::endl;
}

MaskGenerator::~MaskGenerator() {
    // Nothing to cleanup
}

cv::Mat MaskGenerator::thresholdResponse(const cv::Mat& response_map) {
    if (response_map.empty()) {
        return cv::Mat();
    }
    
    // Normalize response to [0, 1]
    double min_val, max_val;
    cv::minMaxLoc(response_map, &min_val, &max_val);
    
    cv::Mat normalized;
    if (max_val - min_val > 1e-6) {
        normalized = (response_map - min_val) / (max_val - min_val);
    } else {
        normalized = cv::Mat::zeros(response_map.size(), CV_32F);
    }
    
    // Threshold
    cv::Mat mask;
    cv::threshold(normalized, mask, config_.mask_threshold, 1.0, cv::THRESH_BINARY);
    
    // Convert to uint8
    mask.convertTo(mask, CV_8U, 255.0);
    
    return mask;
}

cv::Mat MaskGenerator::segmentFromFeatures(const cv::Mat& deep_features, 
                                           const cv::Mat& response_hint) {
    // Simplified deep feature-based segmentation
    // In full implementation, this would use a learned segmentation head
    
    if (deep_features.empty()) {
        return thresholdResponse(response_hint);
    }
    
    // Compute feature magnitude across channels
    int C = deep_features.size[0];
    int H = deep_features.size[1];
    int W = deep_features.size[2];
    
    cv::Mat magnitude = cv::Mat::zeros(H, W, CV_32F);
    
    for (int c = 0; c < C; ++c) {
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = deep_features(ranges).clone();
        channel = channel.reshape(1, H);
        
        // Accumulate squared magnitude
        cv::Mat channel_sq;
        cv::pow(channel, 2.0, channel_sq);
        magnitude += channel_sq;
    }
    
    // Take sqrt to get L2 norm
    cv::sqrt(magnitude, magnitude);
    
    // Resize response hint to match feature size
    cv::Mat response_resized;
    if (response_hint.size() != magnitude.size()) {
        cv::resize(response_hint, response_resized, magnitude.size());
    } else {
        response_resized = response_hint.clone();
    }
    
    // Normalize both
    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);
    cv::normalize(response_resized, response_resized, 0, 1, cv::NORM_MINMAX);
    
    // Combine: weighted average
    cv::Mat combined = 0.6 * magnitude + 0.4 * response_resized;
    
    // Threshold
    cv::Mat mask;
    cv::threshold(combined, mask, config_.mask_threshold, 1.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U, 255.0);
    
    return mask;
}

cv::Mat MaskGenerator::refineMask(const cv::Mat& mask) {
    if (mask.empty()) {
        return cv::Mat();
    }
    
    cv::Mat refined = mask.clone();
    
    // Morphological closing (fill small holes)
    cv::morphologyEx(refined, refined, cv::MORPH_CLOSE, morph_kernel_);
    
    // Morphological opening (remove small noise)
    cv::morphologyEx(refined, refined, cv::MORPH_OPEN, morph_kernel_);
    
    // Fill largest connected component
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(refined, labels, stats, centroids);
    
    if (num_labels > 1) {
        // Find largest component (excluding background label 0)
        int max_area = 0;
        int max_label = 1;
        
        for (int i = 1; i < num_labels; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > max_area) {
                max_area = area;
                max_label = i;
            }
        }
        
        // Keep only largest component
        refined = (labels == max_label);
        refined.convertTo(refined, CV_8U, 255);
    }
    
    return refined;
}

float MaskGenerator::getMaskCoverage(const cv::Mat& mask) {
    if (mask.empty()) {
        return 0.0f;
    }
    
    int total_pixels = mask.rows * mask.cols;
    int foreground_pixels = cv::countNonZero(mask);
    
    return static_cast<float>(foreground_pixels) / total_pixels;
}

cv::Mat MaskGenerator::generateFromResponse(const cv::Mat& response_map) {
    cv::Mat mask = thresholdResponse(response_map);
    mask = refineMask(mask);
    return mask;
}

cv::Mat MaskGenerator::generateFromDeepFeatures(const cv::Mat& response_map,
                                                const cv::Mat& deep_features) {
    if (!config_.use_deep_mask) {
        // Fall back to response-only mask
        return generateFromResponse(response_map);
    }
    
    cv::Mat mask = segmentFromFeatures(deep_features, response_map);
    mask = refineMask(mask);
    
    return mask;
}

} // namespace update_csrt
