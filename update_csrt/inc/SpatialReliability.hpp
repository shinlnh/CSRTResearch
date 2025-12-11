#ifndef UPDATE_CSRT_SPATIAL_RELIABILITY_HPP
#define UPDATE_CSRT_SPATIAL_RELIABILITY_HPP

#include <opencv2/opencv.hpp>
#include "Config.hpp"

namespace update_csrt {

/**
 * @brief Spatial Reliability Map Generator
 * 
 * Computes spatial weights w_s(x,y) based on:
 * 1. Spatial consistency of deep features
 * 2. Distance from target center (Gaussian weighting)
 * 3. Temporal stability across frames
 */
class SpatialReliability {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit SpatialReliability(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~SpatialReliability();
    
    /**
     * @brief Compute spatial reliability map
     * @param deep_features Deep CNN features (C x H x W)
     * @param bbox Current bounding box (for center computation)
     * @return Spatial weight map (H x W, float [0, 1])
     */
    cv::Mat compute(const cv::Mat& deep_features, const cv::Rect& bbox);
    
    /**
     * @brief Update spatial weights with temporal smoothing
     * @param new_weights Newly computed weights
     * @return Smoothed weights
     */
    cv::Mat updateTemporal(const cv::Mat& new_weights);
    
    /**
     * @brief Apply spatial weights to features
     * @param features Input features (C x H x W)
     * @param weights Spatial weights (H x W)
     * @return Weighted features
     */
    cv::Mat applyWeights(const cv::Mat& features, const cv::Mat& weights);
    
    /**
     * @brief Reset temporal history
     */
    void reset();

private:
    Config config_;
    cv::Mat prev_weights_;
    bool first_frame_;
    
    /**
     * @brief Compute Gaussian distance weights from center
     */
    cv::Mat computeGaussianWeights(const cv::Size& size, const cv::Point2f& center);
    
    /**
     * @brief Compute feature consistency across spatial locations
     */
    cv::Mat computeFeatureConsistency(const cv::Mat& deep_features);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_SPATIAL_RELIABILITY_HPP
