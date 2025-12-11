#ifndef UPDATE_CSRT_HOG_EXTRACTOR_HPP
#define UPDATE_CSRT_HOG_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include "Config.hpp"

namespace update_csrt {

/**
 * @brief HOG + ColorNames Feature Extractor for Traditional CSRT
 * 
 * Extracts multi-channel features:
 * - HOG: 21 channels (9 orientations * 2 blocks + 3 normalization)
 * - ColorNames: 10 channels (color name probabilities)
 * Total: 31 channels
 */
class HOGExtractor {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit HOGExtractor(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~HOGExtractor();
    
    /**
     * @brief Extract HOG + ColorNames features
     * @param image Input image patch (BGR, 8UC3)
     * @return Feature tensor (31 x H x W)
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
     * @brief Get number of output channels
     */
    int getNumChannels() const { return num_channels_; }

private:
    Config config_;
    int num_channels_;  // 31 total
    int hog_channels_;  // 21 HOG
    int cn_channels_;   // 10 ColorNames
    
    // ColorNames lookup table (BGR → CN probabilities)
    cv::Mat cn_table_;
    
    /**
     * @brief Compute HOG features
     */
    cv::Mat computeHOG(const cv::Mat& image);
    
    /**
     * @brief Compute ColorNames features
     */
    cv::Mat computeColorNames(const cv::Mat& image);
    
    /**
     * @brief Initialize ColorNames lookup table
     */
    void initColorNamesTable();
    
    /**
     * @brief Apply binary mask to feature tensor
     */
    cv::Mat applyMask(const cv::Mat& features, const cv::Mat& mask);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_HOG_EXTRACTOR_HPP
