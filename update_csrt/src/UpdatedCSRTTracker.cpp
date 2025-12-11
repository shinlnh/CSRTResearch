#include "../inc/UpdatedCSRTTracker.hpp"
#include <iostream>

namespace update_csrt {

UpdatedCSRTTracker::UpdatedCSRTTracker(const Config& config)
    : config_(config), initialized_(false), last_alpha_(config.alpha_default), last_psr_(0.0f) {
    
    // Validate configuration
    if (!config_.validate()) {
        throw std::runtime_error("Invalid configuration");
    }
    
    config_.print();
    
    // Initialize components
    feature_extractor_ = std::make_unique<DeepFeatureExtractor>(config_);
    corr_projection_ = std::make_unique<CorrProjection>(config_);
    adaptive_gating_ = std::make_unique<AdaptiveGating>(config_);
    mask_generator_ = std::make_unique<MaskGenerator>(config_);
    spatial_reliability_ = std::make_unique<SpatialReliability>(config_);
    channel_reliability_ = std::make_unique<ChannelReliability>(config_);
    dcf_solver_ = std::make_unique<DCFSolver>(config_);
    rescue_strategy_ = std::make_unique<RescueStrategy>(config_);
    
    std::cout << "UpdatedCSRTTracker initialized successfully" << std::endl;
}

UpdatedCSRTTracker::~UpdatedCSRTTracker() {
    // Unique pointers handle cleanup automatically
}

cv::Mat UpdatedCSRTTracker::extractPatch(const cv::Mat& frame, const cv::Rect& bbox, 
                                         const cv::Size& output_size, float padding) {
    // Compute padded region
    int pad_w = static_cast<int>(bbox.width * padding);
    int pad_h = static_cast<int>(bbox.height * padding);
    
    cv::Rect padded_bbox(
        bbox.x - pad_w / 2,
        bbox.y - pad_h / 2,
        bbox.width + pad_w,
        bbox.height + pad_h
    );
    
    // Clamp to frame boundaries
    int x1 = std::max(0, padded_bbox.x);
    int y1 = std::max(0, padded_bbox.y);
    int x2 = std::min(frame.cols, padded_bbox.x + padded_bbox.width);
    int y2 = std::min(frame.rows, padded_bbox.y + padded_bbox.height);
    
    cv::Rect valid_bbox(x1, y1, x2 - x1, y2 - y1);
    
    // Extract ROI
    cv::Mat patch = frame(valid_bbox).clone();
    
    // Resize to output size
    cv::Mat resized;
    cv::resize(patch, resized, output_size);
    
    return resized;
}

bool UpdatedCSRTTracker::initialize(const cv::Mat& frame, const cv::Rect& bbox) {
    if (frame.empty() || bbox.width <= 0 || bbox.height <= 0) {
        std::cerr << "Invalid frame or bounding box" << std::endl;
        return false;
    }
    
    std::cout << "Initializing tracker with bbox: " << bbox << std::endl;
    
    current_bbox_ = bbox;
    
    // Extract template patch
    template_img_ = extractPatch(frame, bbox, config_.template_size, config_.padding);
    
    if (template_img_.empty()) {
        std::cerr << "Failed to extract template patch" << std::endl;
        return false;
    }
    
    // Extract deep features from template
    template_features_deep_ = feature_extractor_->extract(template_img_);
    
    if (template_features_deep_.empty()) {
        std::cerr << "Failed to extract deep features" << std::endl;
        return false;
    }
    
    // Generate initial Gaussian label
    cv::Mat label = dcf_solver_->createGaussianLabel(
        cv::Size(template_features_deep_.size[2], template_features_deep_.size[1]),
        2.0f
    );
    
    // Generate initial mask from deep features
    last_mask_ = mask_generator_->generateFromDeepFeatures(label, template_features_deep_);
    
    // Solve for initial CSRT filter (with mask constraint)
    h_csrt_ = dcf_solver_->solveWithMask(template_features_deep_, label, last_mask_);
    
    // Project deep features to correlation space
    h_deep_ = corr_projection_->forward(template_features_deep_);
    
    // Initial blending (use default alpha)
    last_alpha_ = config_.alpha_default;
    h_final_ = adaptive_gating_->blendFilters(h_csrt_, h_deep_, last_alpha_);
    
    // Reset components
    spatial_reliability_->reset();
    channel_reliability_->reset();
    rescue_strategy_->reset();
    
    initialized_ = true;
    
    std::cout << "✓ Tracker initialized successfully" << std::endl;
    
    return true;
}

cv::Point UpdatedCSRTTracker::detectTarget(const cv::Mat& search_patch, 
                                          const cv::Mat& search_features) {
    // Apply final filter to get response map
    last_response_map_ = dcf_solver_->applyFilter(h_final_, search_features);
    
    if (last_response_map_.empty()) {
        return cv::Point(-1, -1);
    }
    
    // Find peak
    cv::Point max_loc;
    cv::minMaxLoc(last_response_map_, nullptr, nullptr, nullptr, &max_loc);
    
    // Compute PSR for quality assessment
    last_psr_ = rescue_strategy_->getLastPSR();
    
    return max_loc;
}

cv::Rect UpdatedCSRTTracker::detectionToBBox(const cv::Point& detection, 
                                            const cv::Rect& search_region,
                                            const cv::Size& template_size) {
    // Map detection from feature space to pixel space
    float scale_x = static_cast<float>(search_region.width) / config_.search_size.width;
    float scale_y = static_cast<float>(search_region.height) / config_.search_size.height;
    
    int det_x = static_cast<int>(detection.x * scale_x);
    int det_y = static_cast<int>(detection.y * scale_y);
    
    // Convert to global coordinates (centered on detection)
    cv::Rect bbox(
        search_region.x + det_x - current_bbox_.width / 2,
        search_region.y + det_y - current_bbox_.height / 2,
        current_bbox_.width,
        current_bbox_.height
    );
    
    return bbox;
}

void UpdatedCSRTTracker::updateFilters(const cv::Mat& frame, const cv::Rect& bbox) {
    // Extract new template
    cv::Mat new_template = extractPatch(frame, bbox, config_.template_size, config_.padding);
    
    // Extract deep features
    cv::Mat new_features = feature_extractor_->extract(new_template);
    
    if (new_features.empty()) {
        return;
    }
    
    // Generate Gaussian label
    cv::Mat label = dcf_solver_->createGaussianLabel(
        cv::Size(new_features.size[2], new_features.size[1]),
        2.0f
    );
    
    // Update binary mask
    last_mask_ = mask_generator_->generateFromDeepFeatures(last_response_map_, new_features);
    
    // Update spatial reliability weights
    cv::Mat spatial_weights = spatial_reliability_->compute(new_features, bbox);
    spatial_weights = spatial_reliability_->updateTemporal(spatial_weights);
    
    // Apply spatial weights to features
    cv::Mat weighted_features = spatial_reliability_->applyWeights(new_features, spatial_weights);
    
    // Solve for updated CSRT filter
    cv::Mat h_csrt_new = dcf_solver_->solveWithMask(weighted_features, label, last_mask_);
    
    // Update with learning rate
    h_csrt_ = (1.0f - config_.learning_rate) * h_csrt_ + config_.learning_rate * h_csrt_new;
    
    // Project deep features
    cv::Mat h_deep_new = corr_projection_->forward(new_features);
    h_deep_ = (1.0f - config_.learning_rate) * h_deep_ + config_.learning_rate * h_deep_new;
    
    // Compute channel reliability weights
    cv::Mat channel_weights = channel_reliability_->compute(
        template_features_deep_, new_features, last_response_map_
    );
    channel_weights = channel_reliability_->updateTemporal(channel_weights);
    
    // Apply channel weights
    h_csrt_ = channel_reliability_->applyWeights(h_csrt_, channel_weights);
    h_deep_ = channel_reliability_->applyWeights(h_deep_, channel_weights);
    
    // Compute adaptive alpha
    last_alpha_ = adaptive_gating_->computeAlpha(
        last_response_map_, template_features_deep_, new_features
    );
    
    // Blend filters
    h_final_ = adaptive_gating_->blendFilters(h_csrt_, h_deep_, last_alpha_);
    
    // Update template
    template_img_ = (1.0f - config_.learning_rate) * template_img_ + 
                    config_.learning_rate * new_template;
    template_features_deep_ = (1.0f - config_.learning_rate) * template_features_deep_ + 
                              config_.learning_rate * new_features;
}

bool UpdatedCSRTTracker::track(const cv::Mat& frame, cv::Rect& bbox) {
    if (!initialized_) {
        std::cerr << "Tracker not initialized" << std::endl;
        return false;
    }
    
    if (frame.empty()) {
        std::cerr << "Empty frame" << std::endl;
        return false;
    }
    
    // Define search region (2x the object size)
    cv::Rect search_region(
        current_bbox_.x - current_bbox_.width / 2,
        current_bbox_.y - current_bbox_.height / 2,
        current_bbox_.width * 2,
        current_bbox_.height * 2
    );
    
    // Clamp to frame
    search_region.x = std::max(0, search_region.x);
    search_region.y = std::max(0, search_region.y);
    search_region.width = std::min(frame.cols - search_region.x, search_region.width);
    search_region.height = std::min(frame.rows - search_region.y, search_region.height);
    
    // Extract search patch
    cv::Mat search_patch = extractPatch(frame, search_region, config_.search_size, 0.0f);
    
    // Extract deep features
    cv::Mat search_features = feature_extractor_->extract(search_patch);
    
    if (search_features.empty()) {
        std::cerr << "Failed to extract search features" << std::endl;
        return false;
    }
    
    // Detect target
    cv::Point detection = detectTarget(search_patch, search_features);
    
    if (detection.x < 0 || detection.y < 0) {
        std::cerr << "Detection failed" << std::endl;
        return false;
    }
    
    // Convert to bounding box
    cv::Rect new_bbox = detectionToBBox(detection, search_region, config_.template_size);
    
    // Check for tracking failure
    bool failure = rescue_strategy_->detectFailure(
        last_response_map_, template_features_deep_, search_features
    );
    
    if (failure) {
        // Attempt recovery
        cv::Rect recovered_bbox = rescue_strategy_->attemptRecovery(
            frame, template_img_, search_region
        );
        
        if (!recovered_bbox.empty()) {
            new_bbox = recovered_bbox;
            std::cout << "✓ Tracking recovered" << std::endl;
        } else {
            std::cerr << "⚠ Tracking lost - recovery failed" << std::endl;
            return false;
        }
    }
    
    // Update tracking quality history
    rescue_strategy_->updateHistory(last_psr_);
    
    // Update filters with new observation
    updateFilters(frame, new_bbox);
    
    // Update state
    current_bbox_ = new_bbox;
    bbox = new_bbox;
    
    return true;
}

void UpdatedCSRTTracker::reset() {
    initialized_ = false;
    template_img_ = cv::Mat();
    template_features_deep_ = cv::Mat();
    h_csrt_ = cv::Mat();
    h_deep_ = cv::Mat();
    h_final_ = cv::Mat();
    last_response_map_ = cv::Mat();
    last_mask_ = cv::Mat();
    
    spatial_reliability_->reset();
    channel_reliability_->reset();
    rescue_strategy_->reset();
}

} // namespace update_csrt
