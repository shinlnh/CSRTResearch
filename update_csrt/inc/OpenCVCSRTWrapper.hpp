#pragma once

#include <opencv2/core.hpp>
#include <opencv2/tracking.hpp>
#include <vector>

namespace update_csrt {

/**
 * @brief Wrapper around OpenCV TrackerCSRT to expose internal states
 * 
 * This class wraps OpenCV's CSRT tracker implementation and provides
 * access to internal states needed for the dual-branch architecture:
 * - Response map (for mask generation and blending)
 * - Filter coefficients (h_csrt)
 * - Binary mask (m)
 * 
 * The wrapper maintains a copy of the last computed response map
 * and filter mask, which are used by the deep feature branch.
 */
class OpenCVCSRTWrapper {
public:
    /**
     * @brief Constructor
     * @param use_hog Enable HOG features (default: true)
     * @param use_color_names Enable ColorNames features (default: true)
     * @param use_segmentation Enable segmentation-based mask (default: false for simplicity)
     * @param padding Spatial padding for search region (default: 3.0)
     * @param template_size Template size (default: 200)
     * @param gsl_sigma Gaussian label sigma (default: 1.0)
     * @param filter_lr Filter learning rate (default: 0.02)
     */
    explicit OpenCVCSRTWrapper(
        bool use_hog = true,
        bool use_color_names = true,
        bool use_segmentation = false,
        float padding = 3.0f,
        float template_size = 200.0f,
        float gsl_sigma = 1.0f,
        float filter_lr = 0.02f
    );
    
    ~OpenCVCSRTWrapper();
    
    /**
     * @brief Initialize tracker with first frame and bounding box
     * @param frame First frame (CV_8UC3 BGR image)
     * @param bbox Initial bounding box
     * @return true if initialization successful
     */
    bool initialize(const cv::Mat& frame, const cv::Rect& bbox);
    
    /**
     * @brief Update tracker with new frame
     * @param frame New frame (CV_8UC3 BGR image)
     * @param bbox Output: Updated bounding box
     * @return true if tracking successful
     */
    bool update(const cv::Mat& frame, cv::Rect& bbox);
    
    /**
     * @brief Get the last computed response map
     * 
     * This is the correlation response map from the CSRT filter.
     * Used for:
     * 1. Generating binary mask for deep features
     * 2. Blending with deep response map
     * 
     * @return Response map (CV_32FC1, single channel)
     */
    cv::Mat getResponseMap() const { return last_response_map_.clone(); }
    
    /**
     * @brief Get the binary mask used in CSRT
     * 
     * This mask constrains the filter update (h = m⊙h in ADMM).
     * In the dual-branch architecture, we apply this mask to
     * deep features before correlation projection.
     * 
     * @return Binary mask (CV_32FC1, values in {0, 1})
     */
    cv::Mat getMask() const { return last_mask_.clone(); }
    
    /**
     * @brief Get filter coefficients (h_csrt) in spatial domain
     * 
     * Note: OpenCV CSRT stores filters in frequency domain (Fourier).
     * This method converts them back to spatial domain for analysis.
     * 
     * @return Vector of filter channels (CV_32FC1, one per feature channel)
     */
    std::vector<cv::Mat> getFilterSpatial() const;
    
    /**
     * @brief Get the current object center
     * @return Center point (x, y)
     */
    cv::Point2f getObjectCenter() const { return object_center_; }
    
    /**
     * @brief Get current scale factor
     * @return Scale relative to initial target size
     */
    float getScaleFactor() const { return current_scale_factor_; }
    
    /**
     * @brief Reset tracker state
     */
    void reset();
    
private:
    // OpenCV CSRT tracker instance
    cv::Ptr<cv::tracking::TrackerCSRT> tracker_;
    
    // Cached internal states
    cv::Mat last_response_map_;     // Last correlation response
    cv::Mat last_mask_;              // Binary spatial mask
    cv::Point2f object_center_;      // Current object center
    float current_scale_factor_;     // Current scale
    
    // Parameters
    bool use_segmentation_;
    
    // Helper: Extract response map after update
    void extractResponseMap(const cv::Mat& frame);
    
    // Helper: Extract mask (if segmentation enabled, otherwise default mask)
    void extractMask();
};

} // namespace update_csrt
