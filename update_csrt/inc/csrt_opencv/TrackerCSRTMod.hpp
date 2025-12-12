#pragma once

#include <opencv2/core.hpp>
#include <opencv2/tracking.hpp>
#include <vector>

namespace update_csrt {
namespace csrt_opencv {

/**
 * @brief Modified CSRT tracker with exposed internal states
 * 
 * This is a wrapper around the modified OpenCV TrackerCSRT implementation
 * that exposes response maps, masks, and filter coefficients needed for
 * the dual-branch architecture.
 * 
 * Based on OpenCV contrib tracking module with modifications to:
 * - Cache and expose correlation response map
 * - Expose binary mask used in filter constraint
 * - Expose filter coefficients for analysis
 */
class TrackerCSRTMod {
public:
    /**
     * @brief CSRT Parameters (matching OpenCV TrackerCSRT::Params)
     */
    struct Params {
        bool use_hog;
        bool use_color_names;
        bool use_gray;
        bool use_rgb;
        bool use_channel_weights;
        bool use_segmentation;
        
        std::string window_function;
        float kaiser_alpha;
        float cheb_attenuation;
        
        float padding;
        float template_size;
        float gsl_sigma;
        
        int hog_orientations;
        float hog_clip;
        int hog_channels;
        int num_hog_channels_used;
        
        float filter_lr;
        float weights_lr;
        
        int admm_iterations;
        int histogram_bins;
        float histogram_lr;
        
        int background_ratio;
        int number_of_scales;
        float scale_sigma_factor;
        float scale_model_max_area;
        float scale_lr;
        float scale_step;
        
        float psr_threshold;
        
        /**
         * @brief Default constructor with CSRT default parameters
         */
        Params();
    };
    
    /**
     * @brief Constructor
     * @param params CSRT parameters
     */
    explicit TrackerCSRTMod(const Params& params = Params());
    
    /**
     * @brief Destructor
     */
    ~TrackerCSRTMod();
    
    /**
     * @brief Initialize tracker
     * @param image First frame (CV_8UC3 BGR)
     * @param boundingBox Initial bounding box
     */
    void init(const cv::Mat& image, const cv::Rect& boundingBox);
    
    /**
     * @brief Update tracker
     * @param image New frame (CV_8UC3 BGR)
     * @param boundingBox Output: updated bounding box
     * @return true if tracking successful
     */
    bool update(const cv::Mat& image, cv::Rect& boundingBox);
    
    /**
     * @brief Set initial mask (optional)
     * @param mask Initial mask (CV_8UC1)
     */
    void setInitialMask(const cv::Mat& mask);
    
    // ========== EXPOSED STATES FOR DUAL-BRANCH ==========
    
    /**
     * @brief Get last correlation response map
     * @return Response map (CV_32FC1)
     */
    cv::Mat getResponseMap() const;
    
    /**
     * @brief Get binary mask (m in h=m⊙h constraint)
     * @return Binary mask (CV_32FC1, values 0 or 1)
     */
    cv::Mat getMask() const;
    
    /**
     * @brief Get filter coefficients in frequency domain
     * @return Vector of filter channels (CV_32FC2 complex)
     */
    std::vector<cv::Mat> getFilterFrequency() const;
    
    /**
     * @brief Get filter coefficients in spatial domain
     * @return Vector of filter channels (CV_32FC1)
     */
    std::vector<cv::Mat> getFilterSpatial() const;
    
    /**
     * @brief Get current object center
     * @return Center point (x, y)
     */
    cv::Point2f getObjectCenter() const;
    
    /**
     * @brief Get current scale factor
     * @return Scale relative to initial target
     */
    float getScaleFactor() const;
    
    /**
     * @brief Get window function
     * @return Window (CV_32FC1)
     */
    cv::Mat getWindow() const;
    
    /**
     * @brief Get HOG cell size
     * @return Cell size in pixels
     */
    int getCellSize() const;
    
private:
    class Impl;  // Forward declaration (pImpl idiom)
    Impl* impl_;  // Pointer to implementation
};

} // namespace csrt_opencv
} // namespace update_csrt
