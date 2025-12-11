#ifndef UPDATE_CSRT_TRACKER_HPP
#define UPDATE_CSRT_TRACKER_HPP

#include <opencv2/opencv.hpp>
#include "Config.hpp"
#include "HOGExtractor.hpp"
#include "DeepFeatureExtractor.hpp"
#include "CorrProjection.hpp"
#include "AdaptiveGating.hpp"
#include "MaskGenerator.hpp"
#include "DCFSolver.hpp"

namespace update_csrt {

/**
 * @brief Updated CSRT Tracker with Deep Features
 * 
 * Dual-branch architecture:
 * 1. Traditional CSRT branch: HOG → DCF → h_csrt
 * 2. Deep feature branch: VGG16 → CorrProject → h_deep
 * 3. Adaptive blending: h_final = α·h_csrt + (1-α)·h_deep
 * 
 * Key innovation: Apply binary mask m to deep features BEFORE projection:
 *   f_masked = f ⊙ m  (element-wise multiplication)
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
    
    // ========== Components ==========
    
    // Traditional CSRT branch
    std::unique_ptr<HOGExtractor> hog_extractor_;
    
    // Deep feature branch
    std::unique_ptr<DeepFeatureExtractor> deep_extractor_;
    std::unique_ptr<CorrProjection> corr_projection_;
    
    // Shared components
    std::unique_ptr<AdaptiveGating> adaptive_gating_;
    std::unique_ptr<MaskGenerator> mask_generator_;
    std::unique_ptr<DCFSolver> dcf_solver_;
    
    // ========== State variables ==========
    
    cv::Rect current_bbox_;
    cv::Mat template_img_;
    
    // CSRT branch state
    cv::Mat template_features_hog_;   // HOG+CN features (31 x H x W)
    cv::Mat h_csrt_;                  // Traditional CSRT filter
    
    // Deep branch state
    cv::Mat template_features_vgg_;   // VGG16 features (512 x H x W)
    cv::Mat h_deep_;                  // Projected deep filter
    
    // Combined state
    cv::Mat h_final_;                 // Blended filter: α·h_csrt + (1-α)·h_deep
    cv::Mat last_mask_;               // Binary mask from CSRT response
    
    // Visualization
    cv::Mat last_response_map_csrt_;  // CSRT branch response
    cv::Mat last_response_map_deep_;  // Deep branch response
    cv::Mat last_response_map_;       // Final blended response
    float last_alpha_;
    float last_psr_;
    
    // ========== Helper methods ==========
    
    /**
     * @brief Extract template patch from frame
     */
    cv::Mat extractPatch(const cv::Mat& frame, const cv::Rect& bbox, 
                         const cv::Size& output_size, float padding);
    
    /**
     * @brief Update both CSRT and deep filters
     */
    void updateFilters(const cv::Mat& frame, const cv::Rect& bbox);
    
    /**
     * @brief Detect target using final blended filter
     */
    cv::Point detectTarget(const cv::Mat& search_patch);
    
    /**
     * @brief Convert detection location to bounding box
     */
    cv::Rect detectionToBBox(const cv::Point& detection, 
                            const cv::Rect& search_region,
                            const cv::Size& template_size);
    
    /**
     * @brief Compute Peak-to-Sidelobe Ratio
     */
    float computePSR(const cv::Mat& response_map);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_TRACKER_HPP
