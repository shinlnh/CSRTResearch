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
 * 
 * KEY: Binary mask is applied at IMAGE-LEVEL before VGG16 to prevent
 * learning background features. This forces VGG to focus only on object region.
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
     * @brief Extract features with binary mask applied at IMAGE LEVEL
     * 
     * ARCHITECTURE CHANGE: Mask is applied to the INPUT IMAGE before VGG16,
     * setting background pixels to mean color. This prevents VGG from learning
     * background features, forcing it to focus only on the object region.
     * 
     * @param image Input image patch (BGR, 8UC3)
     * @param mask Binary mask (CV_32FC1 or CV_8UC1, values 0 or 1/255)
     * @return Feature tensor from object-only region
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
     * @brief Apply binary mask at image level (set background to mean color)
     * 
     * This is the key operation: background pixels are set to ImageNet mean
     * so VGG16 sees them as "neutral" and doesn't learn background features.
     * 
     * @param image Input image
     * @param mask Binary mask (0 = background, 1 or 255 = object)
     * @return Masked image (background = mean color)
     */
    cv::Mat applyMaskToImage(const cv::Mat& image, const cv::Mat& mask);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_DEEP_FEATURE_EXTRACTOR_HPP
