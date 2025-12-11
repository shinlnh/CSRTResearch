#ifndef UPDATE_CSRT_TRACKER_HPP
#define UPDATE_CSRT_TRACKER_HPP

#include <opencv2/opencv.hpp>
#include "Config.hpp"
#include "DeepFeatureExtractor.hpp"
#include "CorrProjection.hpp"
#include "AdaptiveGating.hpp"
#include "MaskGenerator.hpp"
#include "SpatialReliability.hpp"
#include "ChannelReliability.hpp"
#include "DCFSolver.hpp"
#include "RescueStrategy.hpp"

namespace update_csrt {

/**
 * @brief Updated CSRT Tracker with Deep Features
 * 
 * Integrates all components:
 * - Deep feature extraction (VGG16)
 * - CorrProject network for feature projection
 * - Adaptive gating for filter blending
 * - Binary mask generation from deep features
 * - Spatial and channel reliability maps
 * - ADMM-based DCF solver with constraints
 * - Rescue strategy for failure recovery
 */
class UpdatedCSRTTracker {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit UpdatedCSRTTracker(const Config& config = Config());
    
    /**
     * @brief Destructor
     */
    ~UpdatedCSRTTracker();
    
    /**
     * @brief Initialize tracker with first frame and bounding box
     * @param frame First frame (BGR image)
     * @param bbox Initial bounding box
     * @return True if initialization successful
     */
    bool initialize(const cv::Mat& frame, const cv::Rect& bbox);
    
    /**
     * @brief Track object in new frame
     * @param frame New frame (BGR image)
     * @param bbox Output bounding box
     * @return True if tracking successful
     */
    bool track(const cv::Mat& frame, cv::Rect& bbox);
    
    /**
     * @brief Get last response map (for visualization)
     */
    cv::Mat getResponseMap() const { return last_response_map_; }
    
    /**
     * @brief Get last binary mask
     */
    cv::Mat getLastMask() const { return last_mask_; }
    
    /**
     * @brief Get last alpha value
     */
    float getLastAlpha() const { return last_alpha_; }
    
    /**
     * @brief Get last PSR value
     */
    float getLastPSR() const { return last_psr_; }
    
    /**
     * @brief Check if tracker is initialized
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Reset tracker
     */
    void reset();

private:
    Config config_;
    bool initialized_;
    
    // Components
    std::unique_ptr<DeepFeatureExtractor> feature_extractor_;
    std::unique_ptr<CorrProjection> corr_projection_;
    std::unique_ptr<AdaptiveGating> adaptive_gating_;
    std::unique_ptr<MaskGenerator> mask_generator_;
    std::unique_ptr<SpatialReliability> spatial_reliability_;
    std::unique_ptr<ChannelReliability> channel_reliability_;
    std::unique_ptr<DCFSolver> dcf_solver_;
    std::unique_ptr<RescueStrategy> rescue_strategy_;
    
    // State variables
    cv::Rect current_bbox_;
    cv::Mat template_img_;
    cv::Mat template_features_deep_;
    cv::Mat template_features_csrt_;
    cv::Mat h_csrt_;   // Traditional CSRT filter
    cv::Mat h_deep_;   // Deep feature filter (projected)
    cv::Mat h_final_;  // Blended filter
    
    // Visualization
    cv::Mat last_response_map_;
    cv::Mat last_mask_;
    float last_alpha_;
    float last_psr_;
    
    /**
     * @brief Extract template patch from frame
     */
    cv::Mat extractPatch(const cv::Mat& frame, const cv::Rect& bbox, 
                         const cv::Size& output_size, float padding);
    
    /**
     * @brief Update filters with new training sample
     */
    void updateFilters(const cv::Mat& frame, const cv::Rect& bbox);
    
    /**
     * @brief Detect target in search region
     */
    cv::Point detectTarget(const cv::Mat& search_patch, 
                          const cv::Mat& search_features);
    
    /**
     * @brief Convert detection location to bounding box
     */
    cv::Rect detectionToBBox(const cv::Point& detection, 
                            const cv::Rect& search_region,
                            const cv::Size& template_size);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_TRACKER_HPP
