#pragma once

#include <opencv2/opencv.hpp>

#include <deque>
#include <string>
#include <vector>

#include "csrt_scale.hpp"
#include "csrt_segmentation.hpp"

namespace csrt {

struct CsrtParams {
    bool use_channel_weights = true;
    bool use_segmentation = false;
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

    bool use_kf = true;
    int kf_mode = 1;  // 1: KF prior only, 2: PSR gate, 3: PSR+innovation gate
    int kf_prior_mode = 1;  // 0: always use KF prior, 1: use KF prior when prev PSR is low
    int model_lr_mode = 0;  // 0: always, 1: PSR hard, 2: PSR soft, 3: PSR+innovation
    int apce_window = 30;
    float apce_gate_threshold = 0.2f;  // Legacy: normalized APCE gate (unused in paper mode)
    float apce_beta = 0.7f;            // APCE reliability threshold (beta * mean)
    float apce_delta = 0.5f;           // APCE occlusion threshold (delta * mean)
    float apce_redetect_ratio = 0.5f;  // Redetect if APCE > init_apce * ratio
    float apce_redetect_lr_scale = 1.5f;  // Boost learning rate during redetect
    float apce_eps = 1e-6f;
    float apce_norm_eps = 1e-5f;
    int kf_trace_window = 30;
    float kf_r_min = 0.5f;
    float kf_r_max = 5.0f;
    float kf_q_pos = 1e-2f;
    float kf_q_vel = 1e-4f;
    float kf_p_init = 10.0f;
    float kf_innov_base = 5.0f;
    float kf_innov_scale = 0.3f;
    float kf_innov_apce_scale = 0.0f;
    float kf_innov_r_scale = 10.0f;
    float kf_innov_hard_scale = 1.5f;
    float kf_reject_weight = 0.2f;
    float search_scale_min = 1.0f;
    float search_scale_max = 1.1f;  // Tighter adaptive search scale
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

    float GetLastApce() const { return last_apce_; }

    float GetLastApceNorm() const { return last_apce_norm_; }

    float GetLastKfTrace() const { return last_kf_trace_; }

    float GetLastKfUncert() const { return last_kf_uncert_; }

    float GetLastKfR() const { return last_kf_r_; }

    bool GetLastMeasurementAccepted() const { return last_measurement_accepted_; }

    const cv::Point2f &GetLastMeasuredCenter() const { return last_measured_center_; }

    const cv::Point2f &GetLastKfPredCenter() const { return last_kf_pred_center_; }

    const cv::Point2f &GetLastKfCorrectedCenter() const { return last_kf_corrected_center_; }

    float GetLastKfInnov() const { return last_kf_innov_; }

    float GetLastKfInnovThresh() const { return last_kf_innov_thresh_; }

    float GetSearchScaleFactor() const { return search_scale_factor_; }

    const CsrtParams &GetParams() const { return params_; }

private:
    cv::Mat CalculateResponse(const cv::Mat &image, const std::vector<cv::Mat> &filter);
    void UpdateCsrFilter(const cv::Mat &image, const cv::Mat &mask, float lr_scale);
    void UpdateHistograms(const cv::Mat &image, const cv::Rect &region, float lr_scale);
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
    float last_apce_ = 0.0f;
    float last_apce_norm_ = 0.0f;
    float last_apce_mean_ = 0.0f;
    float init_apce_ = 0.0f;
    bool last_apce_valid_ = false;
    float last_kf_trace_ = 0.0f;
    float last_kf_uncert_ = 0.0f;
    float last_kf_r_ = 0.0f;
    float last_kf_innov_ = 0.0f;
    float last_kf_innov_thresh_ = 0.0f;
    bool last_measurement_accepted_ = true;
    cv::Point2f last_measured_center_{0.0f, 0.0f};
    cv::Point2f last_kf_pred_center_{0.0f, 0.0f};
    cv::Point2f last_kf_corrected_center_{0.0f, 0.0f};

    cv::KalmanFilter kf_;
    bool kf_initialized_ = false;
    float search_scale_factor_ = 1.0f;
    std::deque<float> apce_history_;
    std::deque<float> kf_trace_history_;
};

}  // namespace csrt
