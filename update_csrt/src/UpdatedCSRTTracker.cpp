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
    
    std::cout << "\n=== Initializing Dual-Branch Tracker ===" << std::endl;
    
    // ========== Branch 1: Traditional CSRT (HOG) ==========
    std::cout << "Branch 1: Traditional CSRT with HOG+ColorNames..." << std::endl;
    hog_extractor_ = std::make_unique<HOGExtractor>(config_);
    
    // ========== Branch 2: Deep Features (VGG16) ==========
    std::cout << "Branch 2: Deep features with VGG16..." << std::endl;
    deep_extractor_ = std::make_unique<DeepFeatureExtractor>(config_);
    corr_projection_ = std::make_unique<CorrProjection>(config_);
    
    // ========== Shared Components ==========
    std::cout << "Shared components..." << std::endl;
    adaptive_gating_ = std::make_unique<AdaptiveGating>(config_);
    mask_generator_ = std::make_unique<MaskGenerator>(config_);
    dcf_solver_ = std::make_unique<DCFSolver>(config_);
    
    std::cout << "✓ UpdatedCSRTTracker initialized successfully\n" << std::endl;
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

float UpdatedCSRTTracker::computePSR(const cv::Mat& response_map) {
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

bool UpdatedCSRTTracker::initialize(const cv::Mat& frame, const cv::Rect& bbox) {
    if (frame.empty() || bbox.width <= 0 || bbox.height <= 0) {
        std::cerr << "Invalid frame or bounding box" << std::endl;
        return false;
    }
    
    std::cout << "\n=== Initializing Tracker ===" << std::endl;
    std::cout << "Bbox: " << bbox << std::endl;
    
    current_bbox_ = bbox;
    
    // Extract template patch
    template_img_ = extractPatch(frame, bbox, config_.template_size, config_.padding);
    
    if (template_img_.empty()) {
        std::cerr << "Failed to extract template patch" << std::endl;
        return false;
    }
    
    // ========== Branch 1: Extract HOG features ==========
    std::cout << "Extracting HOG+ColorNames features..." << std::endl;
    template_features_hog_ = hog_extractor_->extract(template_img_);
    
    if (template_features_hog_.empty()) {
        std::cerr << "Failed to extract HOG features" << std::endl;
        return false;
    }
    
    // ========== Branch 2: Extract VGG16 features ==========
    std::cout << "Extracting VGG16 deep features..." << std::endl;
    template_features_vgg_ = deep_extractor_->extract(template_img_);
    
    if (template_features_vgg_.empty()) {
        std::cerr << "Failed to extract VGG16 features" << std::endl;
        return false;
    }
    
    // ========== Generate initial Gaussian label ==========
    cv::Mat label = dcf_solver_->createGaussianLabel(
        cv::Size(template_features_hog_.size[2], template_features_hog_.size[1]),
        2.0f
    );
    
    // ========== Generate initial mask from response ==========
    last_mask_ = mask_generator_->generateFromResponse(label);
    
    // ========== Branch 1: Train CSRT filter ==========
    std::cout << "Training CSRT filter (HOG)..." << std::endl;
    h_csrt_ = dcf_solver_->solveWithMask(template_features_hog_, label, last_mask_);
    
    // ========== Branch 2: Train deep filter ==========
    std::cout << "Training deep filter (VGG16 → CorrProject)..." << std::endl;
    
    // Apply mask to IMAGE before VGG16 (background → mean color)
    // KEY: VGG16 only sees object region, doesn't learn background!
    cv::Mat deep_masked = deep_extractor_->extractMasked(template_img_, last_mask_);
    
    // Project to correlation filter space
    h_deep_ = corr_projection_->forward(deep_masked);
    
    // ========== Initial blending ==========
    last_alpha_ = config_.alpha_default;
    h_final_ = adaptive_gating_->blendFilters(h_csrt_, h_deep_, last_alpha_);
    
    initialized_ = true;
    
    std::cout << "✓ Tracker initialized successfully" << std::endl;
    std::cout << "  HOG features: " << template_features_hog_.size << std::endl;
    std::cout << "  VGG features: " << template_features_vgg_.size << std::endl;
    std::cout << "  h_csrt: " << h_csrt_.size() << std::endl;
    std::cout << "  h_deep: " << h_deep_.size() << std::endl;
    std::cout << "  Initial alpha: " << last_alpha_ << "\n" << std::endl;
    
    return true;
}

cv::Point UpdatedCSRTTracker::detectTarget(const cv::Mat& search_patch) {
    // ========== Branch 1: CSRT response ==========
    cv::Mat search_hog = hog_extractor_->extract(search_patch);
    last_response_map_csrt_ = dcf_solver_->applyFilter(h_csrt_, search_hog);
    
    // ========== Branch 2: Deep response ==========
    cv::Mat search_vgg = deep_extractor_->extract(search_patch);
    
    // Apply mask to IMAGE before VGG16 (object-only extraction)
    cv::Mat search_vgg_masked = deep_extractor_->extractMasked(search_patch, last_mask_);
    
    // Project
    cv::Mat search_deep_projected = corr_projection_->forward(search_vgg_masked);
    last_response_map_deep_ = dcf_solver_->applyFilter(h_deep_, search_deep_projected);
    
    // ========== Compute adaptive alpha ==========
    last_alpha_ = adaptive_gating_->computeAlpha(
        last_response_map_csrt_, template_features_hog_, search_hog
    );
    
    // ========== Blend responses ==========
    // Response_final = α·Response_csrt + (1-α)·Response_deep
    if (!last_response_map_csrt_.empty() && !last_response_map_deep_.empty()) {
        cv::addWeighted(last_response_map_csrt_, last_alpha_, 
                        last_response_map_deep_, 1.0 - last_alpha_, 
                        0.0, last_response_map_);
    } else {
        last_response_map_ = last_response_map_csrt_.clone();
    }
    
    // ========== Find peak ==========
    cv::Point max_loc;
    cv::minMaxLoc(last_response_map_, nullptr, nullptr, nullptr, &max_loc);
    
    // Compute PSR
    last_psr_ = computePSR(last_response_map_);
    
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
    
    // ========== Branch 1: Update CSRT filter ==========
    cv::Mat new_hog = hog_extractor_->extract(new_template);
    
    if (!new_hog.empty()) {
        // Update mask from CSRT response
        last_mask_ = mask_generator_->generateFromResponse(last_response_map_csrt_);
        
        // Generate Gaussian label
        cv::Mat label = dcf_solver_->createGaussianLabel(
            cv::Size(new_hog.size[2], new_hog.size[1]),
            2.0f
        );
        
        // Solve for updated CSRT filter
        cv::Mat h_csrt_new = dcf_solver_->solveWithMask(new_hog, label, last_mask_);
        
        // Update with learning rate
        h_csrt_ = (1.0f - config_.learning_rate) * h_csrt_ + config_.learning_rate * h_csrt_new;
    }
    
    // ========== Branch 2: Update deep filter ==========
    cv::Mat new_vgg = deep_extractor_->extract(new_template);
    
    if (!new_vgg.empty()) {
        // Apply mask to IMAGE before VGG16 (background suppression)
        cv::Mat new_vgg_masked = deep_extractor_->extractMasked(new_template, last_mask_);
        
        // Project
        cv::Mat h_deep_new = corr_projection_->forward(new_vgg_masked);
        
        // Update with learning rate
        h_deep_ = (1.0f - config_.learning_rate) * h_deep_ + config_.learning_rate * h_deep_new;
    }
    
    // ========== Blend filters ==========
    h_final_ = adaptive_gating_->blendFilters(h_csrt_, h_deep_, last_alpha_);
    
    // ========== Update templates ==========
    template_img_ = (1.0f - config_.learning_rate) * template_img_ + 
                    config_.learning_rate * new_template;
    
    if (!new_hog.empty()) {
        template_features_hog_ = (1.0f - config_.learning_rate) * template_features_hog_ + 
                                 config_.learning_rate * new_hog;
    }
    
    if (!new_vgg.empty()) {
        template_features_vgg_ = (1.0f - config_.learning_rate) * template_features_vgg_ + 
                                 config_.learning_rate * new_vgg;
    }
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
    
    // ========== Define search region (2x the object size) ==========
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
    
    // ========== Extract search patch ==========
    cv::Mat search_patch = extractPatch(frame, search_region, config_.search_size, 0.0f);
    
    // ========== Detect target (dual-branch) ==========
    cv::Point detection = detectTarget(search_patch);
    
    if (detection.x < 0 || detection.y < 0) {
        std::cerr << "Detection failed" << std::endl;
        return false;
    }
    
    // ========== Convert to bounding box ==========
    cv::Rect new_bbox = detectionToBBox(detection, search_region, config_.template_size);
    
    // ========== Check tracking quality ==========
    if (last_psr_ < 5.0f && config_.verbose) {
        std::cout << "⚠ Low confidence tracking (PSR=" << last_psr_ << ")" << std::endl;
    }
    
    // ========== Update filters ==========
    updateFilters(frame, new_bbox);
    
    // ========== Update state ==========
    current_bbox_ = new_bbox;
    bbox = new_bbox;
    
    if (config_.verbose) {
        std::cout << "α=" << last_alpha_ 
                  << " PSR=" << last_psr_ 
                  << " bbox=" << bbox << std::endl;
    }
    
    return true;
}

void UpdatedCSRTTracker::reset() {
    initialized_ = false;
    template_img_ = cv::Mat();
    template_features_hog_ = cv::Mat();
    template_features_vgg_ = cv::Mat();
    h_csrt_ = cv::Mat();
    h_deep_ = cv::Mat();
    h_final_ = cv::Mat();
    last_response_map_ = cv::Mat();
    last_response_map_csrt_ = cv::Mat();
    last_response_map_deep_ = cv::Mat();
    last_mask_ = cv::Mat();
}

} // namespace update_csrt
