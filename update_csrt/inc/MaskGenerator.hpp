#ifndef UPDATE_CSRT_MASK_GENERATOR_HPP
#define UPDATE_CSRT_MASK_GENERATOR_HPP

#include <opencv2/opencv.hpp>
#include "Config.hpp"

namespace update_csrt {

/**
 * @brief Binary Mask Generator from Deep Features + Response Map
 * 
 * Generates binary foreground/background mask m ∈ {0, 1} using:
 * 1. Response map thresholding
 * 2. Deep feature-based segmentation
 * 3. Morphological operations for refinement
 */
class MaskGenerator {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit MaskGenerator(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~MaskGenerator();
    
    /**
     * @brief Generate binary mask from response map
     * @param response_map Correlation response (H x W, float)
     * @return Binary mask (H x W, 0 or 255)
     */
    cv::Mat generateFromResponse(const cv::Mat& response_map);
    
    /**
     * @brief Generate mask from response + deep features (semantic segmentation)
     * @param response_map Correlation response
     * @param deep_features Deep CNN features (C x H x W)
     * @return Binary mask (H x W, 0 or 255)
     */
    cv::Mat generateFromDeepFeatures(const cv::Mat& response_map,
                                     const cv::Mat& deep_features);
    
    /**
     * @brief Refine mask with morphological operations
     * @param mask Input binary mask
     * @return Refined mask
     */
    cv::Mat refineMask(const cv::Mat& mask);
    
    /**
     * @brief Get mask coverage ratio (percentage of foreground pixels)
     */
    float getMaskCoverage(const cv::Mat& mask);

private:
    Config config_;
    cv::Mat morph_kernel_;
    
    /**
     * @brief Threshold response map to create initial mask
     */
    cv::Mat thresholdResponse(const cv::Mat& response_map);
    
    /**
     * @brief Use deep features for semantic foreground/background separation
     */
    cv::Mat segmentFromFeatures(const cv::Mat& deep_features, 
                                 const cv::Mat& response_hint);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_MASK_GENERATOR_HPP
