#ifndef UPDATE_CSRT_CHANNEL_RELIABILITY_HPP
#define UPDATE_CSRT_CHANNEL_RELIABILITY_HPP

#include <opencv2/opencv.hpp>
#include "Config.hpp"

namespace update_csrt {

/**
 * @brief Channel Reliability Weights Generator
 * 
 * Computes per-channel importance weights w_c based on:
 * 1. Channel discriminative power (from deep features)
 * 2. Temporal stability
 * 3. L2 regularization
 */
class ChannelReliability {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit ChannelReliability(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~ChannelReliability();
    
    /**
     * @brief Compute channel reliability weights
     * @param template_feat Template features (C x H x W)
     * @param search_feat Search features (C x H x W)
     * @param response_map Current response map (H x W)
     * @return Channel weights (C x 1, float [0, 1])
     */
    cv::Mat compute(const cv::Mat& template_feat,
                    const cv::Mat& search_feat,
                    const cv::Mat& response_map);
    
    /**
     * @brief Update channel weights with temporal smoothing
     * @param new_weights Newly computed weights
     * @return Smoothed weights
     */
    cv::Mat updateTemporal(const cv::Mat& new_weights);
    
    /**
     * @brief Apply channel weights to filter
     * @param filter DCF filter (C x H x W)
     * @param weights Channel weights (C x 1)
     * @return Weighted filter
     */
    cv::Mat applyWeights(const cv::Mat& filter, const cv::Mat& weights);
    
    /**
     * @brief Reset temporal history
     */
    void reset();
    
    /**
     * @brief Get current channel weights
     */
    cv::Mat getWeights() const { return current_weights_; }

private:
    Config config_;
    cv::Mat current_weights_;
    cv::Mat prev_weights_;
    bool first_frame_;
    
    /**
     * @brief Compute channel discriminative power
     */
    cv::Mat computeDiscriminativePower(const cv::Mat& template_feat,
                                       const cv::Mat& search_feat);
    
    /**
     * @brief Compute channel contribution to response peak
     */
    cv::Mat computeResponseContribution(const cv::Mat& template_feat,
                                        const cv::Mat& search_feat,
                                        const cv::Mat& response_map);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_CHANNEL_RELIABILITY_HPP
