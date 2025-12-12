/**
 * @file csrt_wrapper.hpp
 * @brief Helper functions to access internal CSRT tracker state for dual-branch blending
 */

#ifndef CSRT_WRAPPER_HPP
#define CSRT_WRAPPER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

namespace cv {
namespace tracking {
namespace impl {

/** @brief Get correlation response map from TrackerCSRT */
cv::Mat getTrackerCSRTResponse(cv::Ptr<cv::TrackerCSRT> tracker);

/** @brief Get binary mask from TrackerCSRT */
cv::Mat getTrackerCSRTMask(cv::Ptr<cv::TrackerCSRT> tracker);

}}} // namespace

#endif // CSRT_WRAPPER_HPP
