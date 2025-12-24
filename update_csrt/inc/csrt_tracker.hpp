#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "csrt_scale.hpp"
#include "csrt_segmentation.hpp"

namespace csrt {

struct CsrtParams {
    bool use_channel_weights = true;
    bool use_segmentation = true;
    bool use_hog = true;
    bool use_color_names = true;
    bool use_gray = true;
    bool use_rgb = false;

    std::string window_function = "hann";
    float kaiser_alpha = 3.75f;
    float cheb_attenuation = 45.0f;
    float padding = 3.0f;
    int template_size = 200;
    float gsl_sigma = 1.0f;

    int hog_orientations = 9;
    float hog_clip = 0.2f;
    int num_hog_channels_used = 18;

    float filter_lr = 0.02f;
    float weights_lr = 0.02f;
    int admm_iterations = 4;

    int number_of_scales = 33;
    float scale_sigma_factor = 0.25f;
    float scale_model_max_area = 512.0f;
    float scale_lr = 0.025f;
    float scale_step = 1.02f;

    int histogram_bins = 16;
    int background_ratio = 2;
    float histogram_lr = 0.04f;

    float psr_threshold = 0.035f;
};

class CsrtTracker {
public:
    explicit CsrtTracker(const CsrtParams &params = CsrtParams());

    void SetInitialMask(const cv::Mat &mask);

    bool Init(const cv::Mat &image, const cv::Rect &bounding_box);

    bool Update(const cv::Mat &image, cv::Rect &bounding_box);

    const cv::Mat &GetResponseMap() const { return last_response_; }

    float GetLastPsr() const { return last_psr_; }

    float GetLastPeak() const { return last_peak_; }

    const CsrtParams &GetParams() const { return params_; }

private:
    cv::Mat CalculateResponse(const cv::Mat &image, const std::vector<cv::Mat> &filter);
    void UpdateCsrFilter(const cv::Mat &image, const cv::Mat &mask);
    void UpdateHistograms(const cv::Mat &image, const cv::Rect &region);
    void ExtractHistograms(const cv::Mat &image, cv::Rect region, Histogram &hf, Histogram &hb);
    std::vector<cv::Mat> CreateCsrFilter(const std::vector<cv::Mat> &img_features,
        const cv::Mat &yf, const cv::Mat &mask);
    cv::Mat GetLocationPrior(const cv::Rect &roi, const cv::Size2f &target_size,
        const cv::Size &image_size);
    cv::Mat SegmentRegion(const cv::Mat &image, const cv::Point2f &object_center,
        const cv::Size2f &template_size, const cv::Size &target_size, float scale_factor);
    cv::Point2f EstimateNewPosition(const cv::Mat &image);
    std::vector<cv::Mat> GetFeatures(const cv::Mat &patch, const cv::Size &feature_size);
    bool CheckMaskArea(const cv::Mat &mask, double object_area) const;
    float ComputePsr(const cv::Mat &response, const cv::Point &peak) const;

    CsrtParams params_;
    float current_scale_factor_ = 1.0f;
    cv::Mat window_;
    cv::Mat yf_;
    cv::Rect2f bounding_box_;
    std::vector<cv::Mat> csr_filter_;
    std::vector<float> filter_weights_;
    cv::Size2f original_target_size_;
    cv::Size image_size_;
    cv::Size2f template_size_;
    cv::Size2i rescaled_template_size_;
    float rescale_ratio_ = 1.0f;
    cv::Point2f object_center_;
    DSST dsst_;
    Histogram hist_foreground_;
    Histogram hist_background_;
    double p_b_ = 0.0;
    cv::Mat erode_element_;
    cv::Mat filter_mask_;
    cv::Mat preset_mask_;
    cv::Mat default_mask_;
    float default_mask_area_ = 0.0f;
    int cell_size_ = 1;

    cv::Mat last_response_;
    float last_psr_ = 0.0f;
    float last_peak_ = 0.0f;
};

}  // namespace csrt
