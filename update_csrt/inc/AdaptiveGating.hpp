#ifndef UPDATE_CSRT_ADAPTIVE_GATING_HPP
#define UPDATE_CSRT_ADAPTIVE_GATING_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "Config.hpp"

namespace update_csrt {

/**
 * @brief Adaptive Gating Network - Learns blending weight α
 * 
 * Takes tracking context as input and outputs adaptive weight α ∈ [0.3, 0.9]
 * for blending h_csrt and h'_deep: h_final = α·h_csrt + (1-α)·h'_deep
 */
class AdaptiveGating {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit AdaptiveGating(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~AdaptiveGating();
    
    /**
     * @brief Compute adaptive blending weight
     * @param response_map Current response map (H x W)
     * @param template_feat Template features (C x H x W)
     * @param search_feat Search features (C x H x W)
     * @return Alpha value ∈ [alpha_min, alpha_max]
     */
    float computeAlpha(const cv::Mat& response_map,
                       const cv::Mat& template_feat,
                       const cv::Mat& search_feat);
    
    /**
     * @brief Blend two filters with adaptive weight
     * @param h_csrt Traditional CSRT filter (C x H x W)
     * @param h_deep Projected deep filter (C x H x W)
     * @param alpha Blending weight
     * @return Blended filter h_final = α·h_csrt + (1-α)·h_deep
     */
    cv::Mat blendFilters(const cv::Mat& h_csrt, 
                         const cv::Mat& h_deep, 
                         float alpha);
    
    /**
     * @brief Check if network is initialized
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Get last computed alpha value
     */
    float getLastAlpha() const { return last_alpha_; }

private:
    Config config_;
    cv::dnn::Net net_;
    bool initialized_;
    float last_alpha_;
    
    /**
     * @brief Extract tracking context features for gating
     */
    cv::Mat extractContextFeatures(const cv::Mat& response_map,
                                    const cv::Mat& template_feat,
                                    const cv::Mat& search_feat);
    
    /**
     * @brief Clamp alpha to valid range
     */
    float clampAlpha(float alpha);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_ADAPTIVE_GATING_HPP
