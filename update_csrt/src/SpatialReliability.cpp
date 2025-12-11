#include "../inc/SpatialReliability.hpp"
#include <iostream>
#include <cmath>

namespace update_csrt {

SpatialReliability::SpatialReliability(const Config& config)
    : config_(config), first_frame_(true) {
    std::cout << "SpatialReliability initialized with " 
              << config_.spatial_bins << " bins" << std::endl;
}

SpatialReliability::~SpatialReliability() {
    // Nothing to cleanup
}

cv::Mat SpatialReliability::computeGaussianWeights(const cv::Size& size, 
                                                   const cv::Point2f& center) {
    cv::Mat weights(size, CV_32F);
    
    float sigma = config_.spatial_sigma * std::min(size.width, size.height);
    float sigma_sq = sigma * sigma;
    
    for (int y = 0; y < size.height; ++y) {
        for (int x = 0; x < size.width; ++x) {
            float dx = x - center.x;
            float dy = y - center.y;
            float dist_sq = dx * dx + dy * dy;
            
            // Gaussian weight: exp(-d²/2σ²)
            weights.at<float>(y, x) = std::exp(-dist_sq / (2.0f * sigma_sq));
        }
    }
    
    return weights;
}

cv::Mat SpatialReliability::computeFeatureConsistency(const cv::Mat& deep_features) {
    if (deep_features.empty()) {
        return cv::Mat();
    }
    
    int C = deep_features.size[0];
    int H = deep_features.size[1];
    int W = deep_features.size[2];
    
    // Compute variance across channels for each spatial location
    cv::Mat consistency = cv::Mat::zeros(H, W, CV_32F);
    
    // First compute mean across channels
    cv::Mat channel_mean = cv::Mat::zeros(H, W, CV_32F);
    for (int c = 0; c < C; ++c) {
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = deep_features(ranges).clone();
        channel = channel.reshape(1, H);
        channel_mean += channel;
    }
    channel_mean /= C;
    
    // Compute variance
    for (int c = 0; c < C; ++c) {
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = deep_features(ranges).clone();
        channel = channel.reshape(1, H);
        
        cv::Mat diff = channel - channel_mean;
        cv::Mat diff_sq;
        cv::pow(diff, 2.0, diff_sq);
        consistency += diff_sq;
    }
    consistency /= C;
    
    // Convert variance to confidence (high variance = high consistency)
    cv::sqrt(consistency, consistency);
    cv::normalize(consistency, consistency, 0, 1, cv::NORM_MINMAX);
    
    return consistency;
}

cv::Mat SpatialReliability::compute(const cv::Mat& deep_features, const cv::Rect& bbox) {
    if (deep_features.empty()) {
        return cv::Mat();
    }
    
    int H = deep_features.size[1];
    int W = deep_features.size[2];
    cv::Size feat_size(W, H);
    
    // Compute center in feature space coordinates
    // Map bbox center to feature dimensions
    cv::Point2f center(W / 2.0f, H / 2.0f);  // Default to center
    
    // Component 1: Gaussian distance weights
    cv::Mat gaussian_weights = computeGaussianWeights(feat_size, center);
    
    cv::Mat spatial_weights;
    
    if (config_.learn_spatial_weights) {
        // Component 2: Feature consistency weights
        cv::Mat feature_weights = computeFeatureConsistency(deep_features);
        
        if (!feature_weights.empty() && feature_weights.size() == gaussian_weights.size()) {
            // Combine both
            spatial_weights = 0.5 * gaussian_weights + 0.5 * feature_weights;
        } else {
            spatial_weights = gaussian_weights;
        }
    } else {
        // Use only Gaussian weights
        spatial_weights = gaussian_weights;
    }
    
    // Normalize to [0, 1]
    cv::normalize(spatial_weights, spatial_weights, 0, 1, cv::NORM_MINMAX);
    
    return spatial_weights;
}

cv::Mat SpatialReliability::updateTemporal(const cv::Mat& new_weights) {
    if (first_frame_) {
        prev_weights_ = new_weights.clone();
        first_frame_ = false;
        return prev_weights_;
    }
    
    // Temporal smoothing: w_t = 0.7·w_t + 0.3·w_{t-1}
    cv::Mat smoothed;
    cv::addWeighted(new_weights, 0.7, prev_weights_, 0.3, 0.0, smoothed);
    
    prev_weights_ = smoothed.clone();
    return smoothed;
}

cv::Mat SpatialReliability::applyWeights(const cv::Mat& features, const cv::Mat& weights) {
    if (features.empty() || weights.empty()) {
        return features;
    }
    
    int C = features.size[0];
    int H = features.size[1];
    int W = features.size[2];
    
    // Resize weights to match feature size if needed
    cv::Mat weights_resized;
    if (weights.size() != cv::Size(W, H)) {
        cv::resize(weights, weights_resized, cv::Size(W, H));
    } else {
        weights_resized = weights;
    }
    
    // Apply weights to each channel
    cv::Mat weighted_features = features.clone();
    
    for (int c = 0; c < C; ++c) {
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = weighted_features(ranges).clone();
        channel = channel.reshape(1, H);
        
        // Multiply by weights
        channel = channel.mul(weights_resized);
        
        // Put back
        channel.reshape(1, 1).copyTo(weighted_features(ranges));
    }
    
    return weighted_features;
}

void SpatialReliability::reset() {
    prev_weights_ = cv::Mat();
    first_frame_ = true;
}

} // namespace update_csrt
