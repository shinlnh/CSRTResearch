#ifndef UPDATE_CSRT_DEEP_FEATURE_EXTRACTOR_HPP
#define UPDATE_CSRT_DEEP_FEATURE_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include "Config.hpp"

namespace update_csrt {

/**
 * @brief Deep Feature Extractor using VGG16 (ONNX Runtime or OpenCV DNN)
 * 
 * Extracts deep convolutional features from VGG16 conv4_3 layer.
 * Features are 512-channel with spatial dimensions depending on input size.
 */
class DeepFeatureExtractor {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit DeepFeatureExtractor(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~DeepFeatureExtractor();
    
    /**
     * @brief Extract deep features from image patch
     * @param image Input image patch (BGR, 8UC3)
     * @return Feature tensor (C x H x W) where C=512 for VGG16 conv4_3
     */
    cv::Mat extract(const cv::Mat& image);
    
    /**
     * @brief Extract features with binary mask applied
     * @param image Input image patch
     * @param mask Binary mask (1-channel, 0 or 255)
     * @return Masked feature tensor
     */
    cv::Mat extractMasked(const cv::Mat& image, const cv::Mat& mask);
    
    /**
     * @brief Get output feature dimensions
     * @return Feature dimensions (channels, height, width)
     */
    cv::Size3i getFeatureSize() const { return feature_size_; }
    
    /**
     * @brief Get number of output channels
     */
    int getNumChannels() const { return feature_size_.x; }
    
    /**
     * @brief Check if extractor is initialized
     */
    bool isInitialized() const { return initialized_; }

private:
    Config config_;
    cv::dnn::Net net_;
    bool initialized_;
    cv::Size3i feature_size_;  // (C, H, W)
    
    // VGG16 ImageNet preprocessing constants
    const cv::Scalar mean_ = cv::Scalar(103.939, 116.779, 123.68); // BGR mean
    const float scale_ = 1.0f;
    
    /**
     * @brief Preprocess image for VGG16 (ImageNet normalization)
     */
    cv::Mat preprocessImage(const cv::Mat& image);
    
    /**
     * @brief Apply binary mask to feature tensor
     */
    cv::Mat applyMaskToFeatures(const cv::Mat& features, const cv::Mat& mask);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_DEEP_FEATURE_EXTRACTOR_HPP
