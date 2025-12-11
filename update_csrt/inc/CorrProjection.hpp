#ifndef UPDATE_CSRT_CORR_PROJECTION_HPP
#define UPDATE_CSRT_CORR_PROJECTION_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "Config.hpp"

namespace update_csrt {

/**
 * @brief CorrProject Network - Projects deep features to correlation filter space
 * 
 * Architecture: Conv2d(512→256) → BN → ReLU → Conv2d(256→64) → BN → ReLU → Conv2d(64→31)
 * Maps 512-channel VGG features to 31-channel correlation filters
 */
class CorrProjection {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit CorrProjection(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~CorrProjection();
    
    /**
     * @brief Project deep features to correlation filter space
     * @param deep_features Input features (512 x H x W)
     * @return Projected features (31 x H x W)
     */
    cv::Mat forward(const cv::Mat& deep_features);
    
    /**
     * @brief Check if network is initialized
     */
    bool isInitialized() const { return initialized_; }

private:
    Config config_;
    cv::dnn::Net net_;
    bool initialized_;
    
    /**
     * @brief Prepare input tensor for network
     */
    cv::Mat prepareInput(const cv::Mat& features);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_CORR_PROJECTION_HPP
