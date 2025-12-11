#include "../inc/RescueStrategy.hpp"
#include <iostream>
#include <numeric>

namespace update_csrt {

RescueStrategy::RescueStrategy(const Config& config)
    : config_(config), in_rescue_mode_(false), consecutive_failures_(0), last_psr_(0.0f) {
    
    quality_history_.resize(config_.rescue_history_size, 1.0f);
    
    std::cout << "RescueStrategy initialized with PSR threshold=" 
              << config_.rescue_threshold << std::endl;
}

RescueStrategy::~RescueStrategy() {
    // Nothing to cleanup
}

float RescueStrategy::computePSR(const cv::Mat& response_map) {
    if (response_map.empty()) {
        return 0.0f;
    }
    
    // Find peak
    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(response_map, nullptr, &max_val, nullptr, &max_loc);
    
    // Create mask excluding peak region (11x11 around peak)
    cv::Mat mask = cv::Mat::ones(response_map.size(), CV_8U);
    int exclude_size = 5;  // ±5 pixels
    
    int y1 = std::max(0, max_loc.y - exclude_size);
    int y2 = std::min(response_map.rows - 1, max_loc.y + exclude_size);
    int x1 = std::max(0, max_loc.x - exclude_size);
    int x2 = std::min(response_map.cols - 1, max_loc.x + exclude_size);
    
    mask(cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)) = 0;
    
    // Compute mean and stddev of sidelobe (excluding peak)
    cv::Scalar mean, stddev;
    cv::meanStdDev(response_map, mean, stddev, mask);
    
    // PSR = (peak - mean) / stddev
    float psr = (max_val - mean[0]) / (stddev[0] + 1e-8);
    
    return psr;
}

float RescueStrategy::computeFeatureSimilarity(const cv::Mat& feat1, const cv::Mat& feat2) {
    if (feat1.empty() || feat2.empty()) {
        return 0.0f;
    }
    
    // Flatten to 1D
    cv::Mat f1_flat = feat1.reshape(1, 1);
    cv::Mat f2_flat = feat2.reshape(1, 1);
    
    // Cosine similarity
    double dot = f1_flat.dot(f2_flat);
    double norm1 = cv::norm(f1_flat);
    double norm2 = cv::norm(f2_flat);
    
    float similarity = static_cast<float>(dot / (norm1 * norm2 + 1e-8));
    
    return similarity;
}

bool RescueStrategy::isQualityDeclining() {
    if (quality_history_.size() < 3) {
        return false;
    }
    
    // Check if last 3 values are declining
    float prev = quality_history_[quality_history_.size() - 3];
    float curr = quality_history_.back();
    
    return (curr < prev * 0.7f);  // 30% drop
}

bool RescueStrategy::detectFailure(const cv::Mat& response_map,
                                  const cv::Mat& template_feat,
                                  const cv::Mat& current_feat) {
    if (!config_.use_rescue) {
        return false;
    }
    
    // Metric 1: Peak-to-Sidelobe Ratio
    float psr = computePSR(response_map);
    last_psr_ = psr;
    
    bool psr_failed = (psr < config_.rescue_threshold);
    
    // Metric 2: Deep feature similarity
    float similarity = computeFeatureSimilarity(template_feat, current_feat);
    bool similarity_failed = (similarity < config_.rescue_similarity_threshold);
    
    // Metric 3: Quality trend
    bool quality_declining = isQualityDeclining();
    
    // Failure if any metric fails
    bool failed = psr_failed || similarity_failed || quality_declining;
    
    if (failed) {
        consecutive_failures_++;
        
        if (consecutive_failures_ >= 3) {
            in_rescue_mode_ = true;
            
            if (config_.verbose) {
                std::cout << "⚠ Tracking failure detected! PSR=" << psr 
                          << ", Similarity=" << similarity << std::endl;
            }
        }
    } else {
        consecutive_failures_ = 0;
        in_rescue_mode_ = false;
    }
    
    return in_rescue_mode_;
}

cv::Rect RescueStrategy::redetectByTemplateMatching(const cv::Mat& frame,
                                                    const cv::Mat& template_img,
                                                    const cv::Rect& search_region) {
    if (frame.empty() || template_img.empty()) {
        return cv::Rect();
    }
    
    // Expand search region by 2x
    cv::Rect expanded_region = search_region;
    int expand_w = search_region.width / 2;
    int expand_h = search_region.height / 2;
    
    expanded_region.x -= expand_w;
    expanded_region.y -= expand_h;
    expanded_region.width += 2 * expand_w;
    expanded_region.height += 2 * expand_h;
    
    // Clamp to frame boundaries
    expanded_region.x = std::max(0, expanded_region.x);
    expanded_region.y = std::max(0, expanded_region.y);
    expanded_region.width = std::min(frame.cols - expanded_region.x, expanded_region.width);
    expanded_region.height = std::min(frame.rows - expanded_region.y, expanded_region.height);
    
    // Extract search ROI
    cv::Mat search_roi = frame(expanded_region);
    
    // Template matching
    cv::Mat result;
    cv::matchTemplate(search_roi, template_img, result, cv::TM_CCOEFF_NORMED);
    
    // Find best match
    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(result, nullptr, &max_val, nullptr, &max_loc);
    
    // Check if match is good enough
    if (max_val > 0.6) {  // Threshold for re-detection
        // Convert to global coordinates
        cv::Rect detected_bbox(
            expanded_region.x + max_loc.x,
            expanded_region.y + max_loc.y,
            template_img.cols,
            template_img.rows
        );
        
        if (config_.verbose) {
            std::cout << "✓ Re-detection successful! Confidence=" << max_val << std::endl;
        }
        
        return detected_bbox;
    }
    
    return cv::Rect();  // Failed
}

cv::Rect RescueStrategy::attemptRecovery(const cv::Mat& frame,
                                        const cv::Mat& template_img,
                                        const cv::Rect& search_region) {
    if (!in_rescue_mode_) {
        return cv::Rect();
    }
    
    if (config_.verbose) {
        std::cout << "Attempting tracking recovery..." << std::endl;
    }
    
    // Try re-detection
    cv::Rect recovered_bbox = redetectByTemplateMatching(frame, template_img, search_region);
    
    if (!recovered_bbox.empty()) {
        // Success! Reset rescue mode
        in_rescue_mode_ = false;
        consecutive_failures_ = 0;
        return recovered_bbox;
    }
    
    // Recovery failed
    return cv::Rect();
}

void RescueStrategy::updateHistory(float quality) {
    quality_history_.push_back(quality);
    
    // Keep only recent history
    while (quality_history_.size() > static_cast<size_t>(config_.rescue_history_size)) {
        quality_history_.pop_front();
    }
}

void RescueStrategy::reset() {
    quality_history_.clear();
    quality_history_.resize(config_.rescue_history_size, 1.0f);
    in_rescue_mode_ = false;
    consecutive_failures_ = 0;
    last_psr_ = 0.0f;
}

} // namespace update_csrt
