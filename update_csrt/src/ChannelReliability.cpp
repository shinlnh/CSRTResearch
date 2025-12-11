#include "../inc/ChannelReliability.hpp"
#include <iostream>

namespace update_csrt {

ChannelReliability::ChannelReliability(const Config& config)
    : config_(config), first_frame_(true) {
    std::cout << "ChannelReliability initialized" << std::endl;
}

ChannelReliability::~ChannelReliability() {
    // Nothing to cleanup
}

cv::Mat ChannelReliability::computeDiscriminativePower(const cv::Mat& template_feat,
                                                       const cv::Mat& search_feat) {
    if (template_feat.empty() || search_feat.empty()) {
        return cv::Mat();
    }
    
    int C = template_feat.size[0];
    cv::Mat weights(C, 1, CV_32F);
    
    // Compute per-channel feature variance (discriminative power)
    for (int c = 0; c < C; ++c) {
        // Extract channel from template
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat temp_channel = template_feat(ranges).clone();
        temp_channel = temp_channel.reshape(1, 1);  // Flatten
        
        // Compute variance
        cv::Scalar mean, stddev;
        cv::meanStdDev(temp_channel, mean, stddev);
        
        // Use stddev as discriminative power (higher variance = more discriminative)
        weights.at<float>(c, 0) = static_cast<float>(stddev[0]);
    }
    
    // Normalize to [0, 1]
    cv::normalize(weights, weights, 0, 1, cv::NORM_MINMAX);
    
    return weights;
}

cv::Mat ChannelReliability::computeResponseContribution(const cv::Mat& template_feat,
                                                        const cv::Mat& search_feat,
                                                        const cv::Mat& response_map) {
    if (template_feat.empty() || search_feat.empty()) {
        return cv::Mat();
    }
    
    int C = template_feat.size[0];
    int H = template_feat.size[1];
    int W = template_feat.size[2];
    
    cv::Mat weights(C, 1, CV_32F, cv::Scalar(0));
    
    // Find peak location in response map
    cv::Point max_loc;
    cv::minMaxLoc(response_map, nullptr, nullptr, nullptr, &max_loc);
    
    // Map response peak to feature coordinates
    int feat_y = max_loc.y * H / response_map.rows;
    int feat_x = max_loc.x * W / response_map.cols;
    feat_y = std::min(feat_y, H - 1);
    feat_x = std::min(feat_x, W - 1);
    
    // Compute per-channel contribution to peak
    for (int c = 0; c < C; ++c) {
        cv::Range temp_ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat temp_channel = template_feat(temp_ranges).clone();
        temp_channel = temp_channel.reshape(1, H);
        
        cv::Range search_ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat search_channel = search_feat(search_ranges).clone();
        search_channel = search_channel.reshape(1, search_feat.size[1]);
        
        // Get values at peak location
        float temp_val = temp_channel.at<float>(feat_y, feat_x);
        
        // Compute similarity between template and search at peak
        cv::Mat temp_flat = temp_channel.reshape(1, 1);
        cv::Mat search_flat = search_channel.reshape(1, 1);
        
        double similarity = temp_flat.dot(search_flat) / 
                           (cv::norm(temp_flat) * cv::norm(search_flat) + 1e-8);
        
        weights.at<float>(c, 0) = static_cast<float>(std::abs(similarity));
    }
    
    // Normalize
    cv::normalize(weights, weights, 0, 1, cv::NORM_MINMAX);
    
    return weights;
}

cv::Mat ChannelReliability::compute(const cv::Mat& template_feat,
                                   const cv::Mat& search_feat,
                                   const cv::Mat& response_map) {
    if (!config_.use_channel_weights) {
        // Return uniform weights
        int C = template_feat.size[0];
        return cv::Mat::ones(C, 1, CV_32F);
    }
    
    cv::Mat weights;
    
    if (config_.learn_channel_weights) {
        // Combine discriminative power and response contribution
        cv::Mat disc_weights = computeDiscriminativePower(template_feat, search_feat);
        cv::Mat resp_weights = computeResponseContribution(template_feat, search_feat, response_map);
        
        if (!disc_weights.empty() && !resp_weights.empty()) {
            weights = 0.5 * disc_weights + 0.5 * resp_weights;
        } else if (!disc_weights.empty()) {
            weights = disc_weights;
        } else {
            int C = template_feat.size[0];
            weights = cv::Mat::ones(C, 1, CV_32F);
        }
    } else {
        // Use only discriminative power
        weights = computeDiscriminativePower(template_feat, search_feat);
    }
    
    // Apply L2 regularization (push towards uniform)
    float reg = config_.channel_reg;
    int C = weights.rows;
    cv::Mat uniform = cv::Mat::ones(C, 1, CV_32F) / C;
    weights = (1.0f - reg) * weights + reg * uniform;
    
    // Normalize
    cv::normalize(weights, weights, 0, 1, cv::NORM_MINMAX);
    
    current_weights_ = weights;
    return weights;
}

cv::Mat ChannelReliability::updateTemporal(const cv::Mat& new_weights) {
    if (first_frame_) {
        prev_weights_ = new_weights.clone();
        first_frame_ = false;
        return prev_weights_;
    }
    
    // Temporal smoothing: w_t = 0.8·w_t + 0.2·w_{t-1}
    cv::Mat smoothed;
    cv::addWeighted(new_weights, 0.8, prev_weights_, 0.2, 0.0, smoothed);
    
    prev_weights_ = smoothed.clone();
    return smoothed;
}

cv::Mat ChannelReliability::applyWeights(const cv::Mat& filter, const cv::Mat& weights) {
    if (filter.empty() || weights.empty()) {
        return filter;
    }
    
    int C = filter.size[0];
    int H = filter.size[1];
    int W = filter.size[2];
    
    if (weights.rows != C) {
        std::cerr << "Channel weight dimension mismatch" << std::endl;
        return filter;
    }
    
    cv::Mat weighted_filter = filter.clone();
    
    // Apply weight to each channel
    for (int c = 0; c < C; ++c) {
        float weight = weights.at<float>(c, 0);
        
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = weighted_filter(ranges).clone();
        channel = channel.reshape(1, H);
        
        // Multiply by weight
        channel *= weight;
        
        // Put back
        channel.reshape(1, 1).copyTo(weighted_filter(ranges));
    }
    
    return weighted_filter;
}

void ChannelReliability::reset() {
    current_weights_ = cv::Mat();
    prev_weights_ = cv::Mat();
    first_frame_ = true;
}

} // namespace update_csrt
