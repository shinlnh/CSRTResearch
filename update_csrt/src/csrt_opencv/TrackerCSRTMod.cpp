#include "../../inc/csrt_opencv/TrackerCSRTMod.hpp"

// Forward declare the TrackerCSRTImpl class from modified source
namespace cv {
inline namespace tracking {
namespace impl {
    class TrackerCSRTImpl;
}
}
}

namespace update_csrt {
namespace csrt_opencv {

// ========== Params Implementation ==========
TrackerCSRTMod::Params::Params()
    : use_hog(true)
    , use_color_names(true)
    , use_gray(true)
    , use_rgb(false)
    , use_channel_weights(true)
    , use_segmentation(false)  // Disabled for simplicity in dual-branch
    , window_function("hann")
    , kaiser_alpha(3.75f)
    , cheb_attenuation(45.0f)
    , padding(3.0f)
    , template_size(200.0f)
    , gsl_sigma(1.0f)
    , hog_orientations(9)
    , hog_clip(0.2f)
    , hog_channels(18)
    , num_hog_channels_used(18)
    , filter_lr(0.02f)
    , weights_lr(0.02f)
    , admm_iterations(4)
    , histogram_bins(16)
    , histogram_lr(0.04f)
    , background_ratio(2)
    , number_of_scales(33)
    , scale_sigma_factor(0.250f)
    , scale_model_max_area(512.0f)
    , scale_lr(0.025f)
    , scale_step(1.020f)
    , psr_threshold(0.035f)
{
}

// ========== pImpl Pattern ==========
class TrackerCSRTMod::Impl {
public:
    // We'll use the modified TrackerCSRTImpl directly
    // This requires linking with the modified trackerCSRT.cpp
    cv::Ptr<cv::tracking::impl::TrackerCSRTImpl> tracker_impl;
    
    Impl(const TrackerCSRTMod::Params& params);
};

TrackerCSRTMod::Impl::Impl(const TrackerCSRTMod::Params& params) {
    // Convert our params to OpenCV TrackerCSRT::Params
    cv::tracking::TrackerCSRT::Params opencv_params;
    
    opencv_params.use_hog = params.use_hog;
    opencv_params.use_color_names = params.use_color_names;
    opencv_params.use_gray = params.use_gray;
    opencv_params.use_rgb = params.use_rgb;
    opencv_params.use_channel_weights = params.use_channel_weights;
    opencv_params.use_segmentation = params.use_segmentation;
    
    opencv_params.window_function = params.window_function;
    opencv_params.kaiser_alpha = params.kaiser_alpha;
    opencv_params.cheb_attenuation = params.cheb_attenuation;
    
    opencv_params.padding = params.padding;
    opencv_params.template_size = params.template_size;
    opencv_params.gsl_sigma = params.gsl_sigma;
    
    opencv_params.hog_orientations = params.hog_orientations;
    opencv_params.hog_clip = params.hog_clip;
    opencv_params.num_hog_channels_used = params.num_hog_channels_used;
    
    opencv_params.filter_lr = params.filter_lr;
    opencv_params.weights_lr = params.weights_lr;
    
    opencv_params.admm_iterations = params.admm_iterations;
    opencv_params.histogram_bins = params.histogram_bins;
    opencv_params.histogram_lr = params.histogram_lr;
    
    opencv_params.background_ratio = params.background_ratio;
    opencv_params.number_of_scales = params.number_of_scales;
    opencv_params.scale_sigma_factor = params.scale_sigma_factor;
    opencv_params.scale_model_max_area = params.scale_model_max_area;
    opencv_params.scale_lr = params.scale_lr;
    opencv_params.scale_step = params.scale_step;
    
    opencv_params.psr_threshold = params.psr_threshold;
    
    // Create the modified tracker implementation
    tracker_impl = cv::makePtr<cv::tracking::impl::TrackerCSRTImpl>(opencv_params);
}

// ========== TrackerCSRTMod Implementation ==========
TrackerCSRTMod::TrackerCSRTMod(const Params& params)
    : impl_(new Impl(params))
{
}

TrackerCSRTMod::~TrackerCSRTMod() {
    delete impl_;
}

void TrackerCSRTMod::init(const cv::Mat& image, const cv::Rect& boundingBox) {
    impl_->tracker_impl->init(image, boundingBox);
}

bool TrackerCSRTMod::update(const cv::Mat& image, cv::Rect& boundingBox) {
    return impl_->tracker_impl->update(image, boundingBox);
}

void TrackerCSRTMod::setInitialMask(const cv::Mat& mask) {
    impl_->tracker_impl->setInitialMask(mask);
}

cv::Mat TrackerCSRTMod::getResponseMap() const {
    return impl_->tracker_impl->getResponseMap();
}

cv::Mat TrackerCSRTMod::getMask() const {
    return impl_->tracker_impl->getMask();
}

std::vector<cv::Mat> TrackerCSRTMod::getFilterFrequency() const {
    return impl_->tracker_impl->getFilterFrequency();
}

std::vector<cv::Mat> TrackerCSRTMod::getFilterSpatial() const {
    return impl_->tracker_impl->getFilterSpatial();
}

cv::Point2f TrackerCSRTMod::getObjectCenter() const {
    return impl_->tracker_impl->getObjectCenter();
}

float TrackerCSRTMod::getScaleFactor() const {
    return impl_->tracker_impl->getScaleFactor();
}

cv::Mat TrackerCSRTMod::getWindow() const {
    return impl_->tracker_impl->getWindow();
}

int TrackerCSRTMod::getCellSize() const {
    return impl_->tracker_impl->getCellSize();
}

} // namespace csrt_opencv
} // namespace update_csrt
