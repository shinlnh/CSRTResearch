#ifndef UPDATE_CSRT_RESCUE_STRATEGY_HPP
#define UPDATE_CSRT_RESCUE_STRATEGY_HPP

#include <opencv2/opencv.hpp>
#include <deque>
#include "Config.hpp"

namespace update_csrt {

/**
 * @brief Rescue Strategy for Tracking Failure Detection and Recovery
 * 
 * Detects tracking failures using:
 * 1. Peak-to-Sidelobe Ratio (PSR) drop
 * 2. Deep feature similarity degradation
 * 3. Response map quality metrics
 * 
 * Recovery strategies:
 * 1. Re-detection using template matching
 * 2. Filter rollback to previous good state
 * 3. Search region expansion
 */
class RescueStrategy {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit RescueStrategy(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~RescueStrategy();
    
    /**
     * @brief Detect if tracking has failed
     * @param response_map Current response map (H x W)
     * @param template_feat Template features (C x H x W)
     * @param current_feat Current frame features (C x H x W)
     * @return True if failure detected
     */
    bool detectFailure(const cv::Mat& response_map,
                       const cv::Mat& template_feat,
                       const cv::Mat& current_feat);
    
    /**
     * @brief Attempt to recover from tracking failure
     * @param frame Current frame
     * @param template_img Template image
     * @param search_region Current search region
     * @return Recovered bounding box (empty if failed)
     */
    cv::Rect attemptRecovery(const cv::Mat& frame,
                            const cv::Mat& template_img,
                            const cv::Rect& search_region);
    
    /**
     * @brief Update tracking quality history
     * @param quality Current quality score (PSR, similarity, etc.)
     */
    void updateHistory(float quality);
    
    /**
     * @brief Check if tracker is currently in rescue mode
     */
    bool isInRescueMode() const { return in_rescue_mode_; }
    
    /**
     * @brief Get last computed PSR value
     */
    float getLastPSR() const { return last_psr_; }
    
    /**
     * @brief Reset rescue state
     */
    void reset();

private:
    Config config_;
    std::deque<float> quality_history_;
    bool in_rescue_mode_;
    int consecutive_failures_;
    float last_psr_;
    
    /**
     * @brief Compute Peak-to-Sidelobe Ratio
     */
    float computePSR(const cv::Mat& response_map);
    
    /**
     * @brief Compute deep feature similarity (cosine)
     */
    float computeFeatureSimilarity(const cv::Mat& feat1, const cv::Mat& feat2);
    
    /**
     * @brief Template matching-based re-detection
     */
    cv::Rect redetectByTemplateMatching(const cv::Mat& frame,
                                        const cv::Mat& template_img,
                                        const cv::Rect& search_region);
    
    /**
     * @brief Check if quality trend is declining
     */
    bool isQualityDeclining();
};

} // namespace update_csrt

#endif // UPDATE_CSRT_RESCUE_STRATEGY_HPP
