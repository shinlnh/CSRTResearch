#include "../inc/csrt_tracker.hpp"

#include "../inc/csrt_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace csrt {

namespace {

float Clamp01(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

void UpdateWindow(std::deque<float> &window, float value, int max_size) {
    if (max_size <= 0) {
        return;
    }
    window.push_back(value);
    if (static_cast<int>(window.size()) > max_size) {
        window.pop_front();
    }
}

float WindowMin(const std::deque<float> &window, float fallback) {
    if (window.empty()) {
        return fallback;
    }
    return *std::min_element(window.begin(), window.end());
}

float WindowMax(const std::deque<float> &window, float fallback) {
    if (window.empty()) {
        return fallback;
    }
    return *std::max_element(window.begin(), window.end());
}

float WindowMean(const std::deque<float> &window, float fallback) {
    if (window.empty()) {
        return fallback;
    }
    float sum = std::accumulate(window.begin(), window.end(), 0.0f);
    return sum / static_cast<float>(window.size());
}

float ComputeApce(const cv::Mat &response, float eps) {
    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(response, &min_val, &max_val, nullptr, nullptr);
    cv::Mat diff = response - min_val;
    cv::Mat diff_sq;
    cv::multiply(diff, diff, diff_sq);
    cv::Scalar mean_val = cv::mean(diff_sq);
    float denom = static_cast<float>(mean_val[0] + eps);
    float num = static_cast<float>((max_val - min_val) * (max_val - min_val));
    if (denom <= 0.0f) {
        return 0.0f;
    }
    return num / denom;
}

}  // namespace

class ParallelCreateCsrFilter : public cv::ParallelLoopBody {
public:
    ParallelCreateCsrFilter(const std::vector<cv::Mat> &img_features, const cv::Mat &yf,
        const cv::Mat &mask, int admm_iterations, std::vector<cv::Mat> &result_filter)
        : img_features_(img_features), yf_(yf), mask_(mask),
          admm_iterations_(admm_iterations), result_filter_(result_filter) {}

    void operator()(const cv::Range &range) const override {
        for (int i = range.start; i < range.end; ++i) {
            float mu = 5.0f;
            float beta = 3.0f;
            float mu_max = 20.0f;
            float lambda = mu / 100.0f;

            const cv::Mat &f = img_features_[i];
            cv::Mat sxy;
            cv::Mat sxx;
            cv::mulSpectrums(f, yf_, sxy, 0, true);
            cv::mulSpectrums(f, f, sxx, 0, true);

            cv::Mat h = DivideComplexMatrices(sxy, (sxx + lambda));
            cv::idft(h, h, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
            h = h.mul(mask_);
            cv::dft(h, h, cv::DFT_COMPLEX_OUTPUT);

            cv::Mat lagrangian = cv::Mat::zeros(h.size(), h.type());
            cv::Mat g;
            for (int iter = 0; iter < admm_iterations_; ++iter) {
                g = DivideComplexMatrices((sxy + (mu * h) - lagrangian), (sxx + mu));
                cv::idft((mu * g) + lagrangian, h, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
                float lm = 1.0f / (lambda + mu);
                h = h.mul(mask_ * lm);
                cv::dft(h, h, cv::DFT_COMPLEX_OUTPUT);

                lagrangian = lagrangian + mu * (g - h);
                mu = std::min(mu_max, beta * mu);
            }
            result_filter_[i] = h;
        }
    }

private:
    std::vector<cv::Mat> img_features_;
    cv::Mat yf_;
    cv::Mat mask_;
    int admm_iterations_;
    std::vector<cv::Mat> &result_filter_;
};

CsrtTracker::CsrtTracker(const CsrtParams &params) : params_(params) {}

void CsrtTracker::SetInitialMask(const cv::Mat &mask) {
    preset_mask_ = mask.clone();
}

bool CsrtTracker::CheckMaskArea(const cv::Mat &mask, double object_area) const {
    double threshold = 0.05;
    double mask_area = cv::sum(mask)[0];
    return mask_area >= threshold * object_area;
}

float CsrtTracker::ComputePsr(const cv::Mat &response, const cv::Point &peak) const {
    const int half_size = 5;
    cv::Rect peak_rect(peak.x - half_size, peak.y - half_size, 2 * half_size + 1, 2 * half_size + 1);
    peak_rect &= cv::Rect(0, 0, response.cols, response.rows);

    cv::Mat mask = cv::Mat::ones(response.size(), CV_8U);
    mask(peak_rect).setTo(0);

    cv::Scalar mean_val;
    cv::Scalar std_val;
    cv::meanStdDev(response, mean_val, std_val, mask);
    double denom = std_val[0];
    if (denom < 1e-6) {
        denom = 1e-6;
    }
    return static_cast<float>((response.at<float>(peak) - mean_val[0]) / denom);
}

cv::Mat CsrtTracker::CalculateResponse(const cv::Mat &image, const std::vector<cv::Mat> &filter) {
    float search_scale = std::max(0.01f, search_scale_factor_);
    int patch_w = std::max(1, cvFloor(current_scale_factor_ * search_scale * template_size_.width));
    int patch_h = std::max(1, cvFloor(current_scale_factor_ * search_scale * template_size_.height));
    cv::Mat patch = GetSubwindow(image, object_center_, patch_w, patch_h);
    if (patch.empty()) {
        return cv::Mat();
    }
    cv::resize(patch, patch, rescaled_template_size_, 0, 0, cv::INTER_CUBIC);

    std::vector<cv::Mat> features = GetFeatures(patch, yf_.size());
    std::vector<cv::Mat> fft_features = FourierTransformFeatures(features);

    cv::Mat response = cv::Mat::zeros(fft_features[0].size(), CV_32FC2);
    if (params_.use_channel_weights) {
        cv::Mat resp_ch;
        for (size_t i = 0; i < fft_features.size(); ++i) {
            cv::mulSpectrums(fft_features[i], filter[i], resp_ch, 0, true);
            response += (resp_ch * filter_weights_[i]);
        }
    } else {
        cv::Mat resp_ch;
        for (size_t i = 0; i < fft_features.size(); ++i) {
            cv::mulSpectrums(fft_features[i], filter[i], resp_ch, 0, true);
            response = response + resp_ch;
        }
    }

    cv::idft(response, response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return response;
}

void CsrtTracker::UpdateCsrFilter(const cv::Mat &image, const cv::Mat &mask, float lr_scale) {
    const float weights_lr = std::min(1.0f, params_.weights_lr * lr_scale);
    const float filter_lr = std::min(1.0f, params_.filter_lr * lr_scale);
    if (weights_lr <= 0.0f && filter_lr <= 0.0f) {
        return;
    }
    cv::Mat patch = GetSubwindow(image, object_center_,
        cvFloor(current_scale_factor_ * template_size_.width),
        cvFloor(current_scale_factor_ * template_size_.height));
    if (patch.empty()) {
        return;
    }
    cv::resize(patch, patch, rescaled_template_size_, 0, 0, cv::INTER_CUBIC);

    std::vector<cv::Mat> features = GetFeatures(patch, yf_.size());
    std::vector<cv::Mat> fft_features = FourierTransformFeatures(features);
    std::vector<cv::Mat> new_csr_filter = CreateCsrFilter(fft_features, yf_, mask);

    if (params_.use_channel_weights) {
        cv::Mat current_resp;
        double max_val = 0.0;
        float sum_weights = 0.0f;
        std::vector<float> new_filter_weights(new_csr_filter.size());
        for (size_t i = 0; i < new_csr_filter.size(); ++i) {
            cv::mulSpectrums(fft_features[i], new_csr_filter[i], current_resp, 0, true);
            cv::idft(current_resp, current_resp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
            cv::minMaxLoc(current_resp, nullptr, &max_val, nullptr, nullptr);
            sum_weights += static_cast<float>(max_val);
            new_filter_weights[i] = static_cast<float>(max_val);
        }

        float updated_sum = 0.0f;
        for (size_t i = 0; i < filter_weights_.size(); ++i) {
            filter_weights_[i] = filter_weights_[i] * (1.0f - weights_lr) +
                weights_lr * (new_filter_weights[i] / sum_weights);
            updated_sum += filter_weights_[i];
        }

        for (size_t i = 0; i < filter_weights_.size(); ++i) {
            filter_weights_[i] /= updated_sum;
        }
    }

    for (size_t i = 0; i < csr_filter_.size(); ++i) {
        csr_filter_[i] = (1.0f - filter_lr) * csr_filter_[i] +
            filter_lr * new_csr_filter[i];
    }
}

std::vector<cv::Mat> CsrtTracker::GetFeatures(const cv::Mat &patch, const cv::Size &feature_size) {
    std::vector<cv::Mat> features;

    if (params_.use_hog) {
        std::vector<cv::Mat> hog = GetFeaturesHog(patch, cell_size_);
        features.insert(features.end(), hog.begin(),
            hog.begin() + std::min(static_cast<size_t>(params_.num_hog_channels_used), hog.size()));
    }

    if (params_.use_color_names) {
        std::vector<cv::Mat> cn = GetFeaturesCn(patch, feature_size);
        features.insert(features.end(), cn.begin(), cn.end());
    }

    if (params_.use_gray) {
        cv::Mat gray;
        cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gray, feature_size, 0, 0, cv::INTER_CUBIC);
        gray.convertTo(gray, CV_32FC1, 1.0 / 255.0, -0.5);
        features.push_back(gray);
    }

    if (params_.use_rgb) {
        std::vector<cv::Mat> rgb = GetFeaturesRgb(patch, feature_size);
        features.insert(features.end(), rgb.begin(), rgb.end());
    }

    for (size_t i = 0; i < features.size(); ++i) {
        features[i] = features[i].mul(window_);
    }

    return features;
}

std::vector<cv::Mat> CsrtTracker::CreateCsrFilter(
    const std::vector<cv::Mat> &img_features, const cv::Mat &yf, const cv::Mat &mask) {
    std::vector<cv::Mat> result_filter(img_features.size());
    ParallelCreateCsrFilter parallel_worker(img_features, yf, mask,
        params_.admm_iterations, result_filter);
    cv::parallel_for_(cv::Range(0, static_cast<int>(result_filter.size())), parallel_worker);
    return result_filter;
}

cv::Mat CsrtTracker::GetLocationPrior(const cv::Rect &roi, const cv::Size2f &target_size,
    const cv::Size &image_size) {
    int x1 = cvRound(std::max(std::min(roi.x - 1, image_size.width - 1), 0));
    int y1 = cvRound(std::max(std::min(roi.y - 1, image_size.height - 1), 0));

    int x2 = cvRound(std::min(std::max(roi.width - 1, 0), image_size.width - 1));
    int y2 = cvRound(std::min(std::max(roi.height - 1, 0), image_size.height - 1));

    cv::Size target_size_sq;
    target_size_sq.width = target_size_sq.height = cvFloor(std::min(target_size.width, target_size.height));

    double cx = x1 + (x2 - x1) / 2.0;
    double cy = y1 + (y2 - y1) / 2.0;
    double kernel_size_width = 1.0 / (0.5 * static_cast<double>(target_size_sq.width) * 1.4142 + 1.0);
    double kernel_size_height = 1.0 / (0.5 * static_cast<double>(target_size_sq.height) * 1.4142 + 1.0);

    cv::Mat kernel_weight = cv::Mat::zeros(1 + cvFloor(y2 - y1),
        1 + cvFloor(-(x1 - cx) + (x2 - cx)), CV_64FC1);
    for (int y = y1; y < y2 + 1; ++y) {
        double *weight_ptr = kernel_weight.ptr<double>(y);
        double tmp_y = std::pow((cy - y) * kernel_size_height, 2);
        for (int x = x1; x < x2 + 1; ++x) {
            weight_ptr[x] = KernelEpan(std::pow((cx - x) * kernel_size_width, 2) + tmp_y);
        }
    }

    double max_val = 0.0;
    cv::minMaxLoc(kernel_weight, nullptr, &max_val, nullptr, nullptr);
    cv::Mat fg_prior = kernel_weight / max_val;
    fg_prior.setTo(0.5, fg_prior < 0.5);
    fg_prior.setTo(0.9, fg_prior > 0.9);
    return fg_prior;
}

cv::Mat CsrtTracker::SegmentRegion(const cv::Mat &image, const cv::Point2f &object_center,
    const cv::Size2f &template_size, const cv::Size &target_size, float scale_factor) {
    cv::Rect valid_pixels;
    cv::Mat patch = GetSubwindow(image, object_center,
        cvFloor(scale_factor * template_size.width),
        cvFloor(scale_factor * template_size.height), &valid_pixels);
    if (patch.empty()) {
        return cv::Mat();
    }

    cv::Size2f scaled_target(target_size.width * scale_factor,
        target_size.height * scale_factor);

    cv::Mat fg_prior = GetLocationPrior(cv::Rect(0, 0, patch.cols, patch.rows),
        scaled_target, patch.size());

    std::vector<cv::Mat> img_channels;
    cv::split(patch, img_channels);

    auto probs = Segment::ComputePosteriors2(img_channels, 0, 0, patch.cols, patch.rows,
        p_b_, fg_prior, 1.0 - fg_prior, hist_foreground_, hist_background_);

    cv::Mat mask = cv::Mat::zeros(probs.first.size(), probs.first.type());
    probs.first(valid_pixels).copyTo(mask(valid_pixels));
    double max_resp = GetMax(mask);
    cv::threshold(mask, mask, max_resp / 2.0, 1.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_32FC1, 1.0);
    return mask;
}

void CsrtTracker::ExtractHistograms(const cv::Mat &image, cv::Rect region,
    Histogram &hf, Histogram &hb) {
    int x1 = std::min(std::max(0, region.x), image.cols - 1);
    int y1 = std::min(std::max(0, region.y), image.rows - 1);
    int x2 = std::min(std::max(0, region.x + region.width), image.cols - 1);
    int y2 = std::min(std::max(0, region.y + region.height), image.rows - 1);

    int offset_x = (x2 - x1 + 1) / params_.background_ratio;
    int offset_y = (y2 - y1 + 1) / params_.background_ratio;
    int outer_y1 = std::max(0, y1 - offset_y);
    int outer_y2 = std::min(image.rows, y2 + offset_y + 1);
    int outer_x1 = std::max(0, x1 - offset_x);
    int outer_x2 = std::min(image.cols, x2 + offset_x + 1);

    p_b_ = 1.0 - ((x2 - x1 + 1) * (y2 - y1 + 1)) /
        (static_cast<double>(outer_x2 - outer_x1 + 1) * (outer_y2 - outer_y1 + 1));

    std::vector<cv::Mat> img_channels(image.channels());
    cv::split(image, img_channels);
    for (size_t k = 0; k < img_channels.size(); ++k) {
        img_channels[k].convertTo(img_channels[k], CV_8UC1);
    }

    hf.ExtractForegroundHistogram(img_channels, cv::Mat(), false, x1, y1, x2, y2);
    hb.ExtractBackgroundHistogram(img_channels, x1, y1, x2, y2,
        outer_x1, outer_y1, outer_x2, outer_y2);
}

void CsrtTracker::UpdateHistograms(const cv::Mat &image, const cv::Rect &region, float lr_scale) {
    const float hist_lr = std::min(1.0f, params_.histogram_lr * lr_scale);
    if (hist_lr <= 0.0f) {
        return;
    }
    Histogram hf(image.channels(), params_.histogram_bins);
    Histogram hb(image.channels(), params_.histogram_bins);
    ExtractHistograms(image, region, hf, hb);

    std::vector<double> hf_new = hf.GetHistogramVector();
    std::vector<double> hb_new = hb.GetHistogramVector();
    std::vector<double> hf_old = hist_foreground_.GetHistogramVector();
    std::vector<double> hb_old = hist_background_.GetHistogramVector();

    for (size_t i = 0; i < hf_new.size(); ++i) {
        hf_new[i] = (1.0 - hist_lr) * hf_old[i] + hist_lr * hf_new[i];
        hb_new[i] = (1.0 - hist_lr) * hb_old[i] + hist_lr * hb_new[i];
    }

    hist_foreground_.SetHistogramVector(hf_new.data());
    hist_background_.SetHistogramVector(hb_new.data());
}

cv::Point2f CsrtTracker::EstimateNewPosition(const cv::Mat &image) {
    cv::Mat response = CalculateResponse(image, csr_filter_);
    last_response_ = response.clone();

    if (response.empty()) {
        last_peak_ = 0.0f;
        last_psr_ = 0.0f;
        last_apce_ = 0.0f;
        last_apce_valid_ = false;
        UpdateWindow(apce_history_, last_apce_, params_.apce_window);
        float apce_min = WindowMin(apce_history_, last_apce_);
        float apce_max = WindowMax(apce_history_, last_apce_);
        last_apce_norm_ = Clamp01((last_apce_ - apce_min) /
            (apce_max - apce_min + params_.apce_norm_eps));
        last_apce_mean_ = WindowMean(apce_history_, last_apce_);
        return object_center_;
    }

    double max_val = 0.0;
    cv::Point max_loc;
    cv::minMaxLoc(response, nullptr, &max_val, nullptr, &max_loc);
    last_peak_ = static_cast<float>(max_val);
    last_psr_ = ComputePsr(response, max_loc);
    last_apce_ = ComputeApce(response, params_.apce_eps);
    last_apce_valid_ = true;
    UpdateWindow(apce_history_, last_apce_, params_.apce_window);
    float apce_min = WindowMin(apce_history_, last_apce_);
    float apce_max = WindowMax(apce_history_, last_apce_);
    last_apce_norm_ = Clamp01((last_apce_ - apce_min) /
        (apce_max - apce_min + params_.apce_norm_eps));
    last_apce_mean_ = WindowMean(apce_history_, last_apce_);
    if (init_apce_ <= 0.0f && last_apce_ > 0.0f) {
        init_apce_ = last_apce_;
    }

    float col = static_cast<float>(max_loc.x) + SubpixelPeak(response, "horizontal", max_loc);
    float row = static_cast<float>(max_loc.y) + SubpixelPeak(response, "vertical", max_loc);

    if (row + 1 > response.rows / 2.0f) {
        row = row - response.rows;
    }
    if (col + 1 > response.cols / 2.0f) {
        col = col - response.cols;
    }

    float search_scale = std::max(0.01f, search_scale_factor_);
    cv::Point2f new_center = object_center_ +
        cv::Point2f(search_scale * current_scale_factor_ * (1.0f / rescale_ratio_) * cell_size_ * col,
            search_scale * current_scale_factor_ * (1.0f / rescale_ratio_) * cell_size_ * row);

    if (new_center.x < 0) {
        new_center.x = 0;
    }
    if (new_center.x >= image_size_.width) {
        new_center.x = static_cast<float>(image_size_.width - 1);
    }
    if (new_center.y < 0) {
        new_center.y = 0;
    }
    if (new_center.y >= image_size_.height) {
        new_center.y = static_cast<float>(image_size_.height - 1);
    }

    return new_center;
}

bool CsrtTracker::Update(const cv::Mat &image, cv::Rect &bounding_box) {
    cv::Mat frame;
    if (image.channels() == 1) {
        cv::cvtColor(image, frame, cv::COLOR_GRAY2BGR);
    } else {
        frame = image;
    }

    const bool has_kf = params_.use_kf && kf_initialized_;
    cv::Point2f kf_pred_center = object_center_;
    last_measured_center_ = cv::Point2f(0.0f, 0.0f);
    last_kf_pred_center_ = kf_pred_center;
    last_kf_corrected_center_ = kf_pred_center;
    last_kf_innov_ = 0.0f;
    last_kf_innov_thresh_ = 0.0f;
    float lr_scale = 1.0f;
    const float prev_psr = last_psr_;

    if (has_kf) {
        cv::Mat pred = kf_.predict();
        kf_pred_center.x = pred.at<float>(0);
        kf_pred_center.y = pred.at<float>(1);
        last_kf_pred_center_ = kf_pred_center;
        bool use_kf_prior = true;
        if (params_.kf_prior_mode == 1 && params_.psr_threshold > 0.0f) {
            use_kf_prior = prev_psr < params_.psr_threshold;
        }
        if (use_kf_prior) {
            object_center_ = kf_pred_center;
        }
    }

    // Estimate position using correlation filter from current position (KF prior if enabled).
    cv::Point2f measured_center = EstimateNewPosition(frame);
    last_measured_center_ = measured_center;

    float psr_conf = 1.0f;
    if (params_.psr_threshold > 0.0f) {
        psr_conf = Clamp01((last_psr_ - params_.psr_threshold) / params_.psr_threshold);
    }
    const bool psr_good = (params_.psr_threshold <= 0.0f) ||
        (last_psr_ >= params_.psr_threshold);

    float innov_thresh = 0.0f;
    float innovation = 0.0f;
    bool innovation_ok = true;
    if (has_kf) {
        const float size_ref = std::max({original_target_size_.width, original_target_size_.height, 1.0f});
        innov_thresh = params_.kf_innov_base + params_.kf_innov_scale * size_ref;
        const float dx = measured_center.x - kf_pred_center.x;
        const float dy = measured_center.y - kf_pred_center.y;
        innovation = std::sqrt(dx * dx + dy * dy);
        last_kf_innov_ = innovation;
        last_kf_innov_thresh_ = innov_thresh * params_.kf_innov_hard_scale;
        innovation_ok = innovation <= last_kf_innov_thresh_;
    }

    bool kf_update = false;
    switch (params_.kf_mode) {
        case 1:
            kf_update = has_kf;
            break;
        case 2:
            kf_update = has_kf && psr_good;
            break;
        case 3:
            kf_update = has_kf && psr_good && innovation_ok;
            break;
        default:
            kf_update = has_kf && psr_good;
            break;
    }

    bool model_update = true;
    switch (params_.model_lr_mode) {
        case 0:
            model_update = true;
            lr_scale = 1.0f;
            break;
        case 1:
            model_update = psr_good;
            lr_scale = model_update ? 1.0f : 0.0f;
            break;
        case 2:
            model_update = psr_conf > 0.0f;
            lr_scale = model_update ? psr_conf : 0.0f;
            break;
        case 3:
            model_update = psr_good && innovation_ok;
            lr_scale = model_update ? 1.0f : 0.0f;
            break;
        default:
            model_update = psr_good;
            lr_scale = model_update ? 1.0f : 0.0f;
            break;
    }

    if (has_kf) {
        float r_t = params_.kf_r_max - psr_conf * (params_.kf_r_max - params_.kf_r_min);
        if (!kf_update) {
            r_t = params_.kf_r_max;
        }
        r_t = std::max(r_t, 1e-6f);
        last_kf_r_ = r_t;
        kf_.measurementNoiseCov = (cv::Mat_<float>(2, 2) << r_t, 0.0f, 0.0f, r_t);

        if (kf_update) {
            cv::Mat meas(2, 1, CV_32F);
            meas.at<float>(0) = measured_center.x;
            meas.at<float>(1) = measured_center.y;
            cv::Mat corrected = kf_.correct(meas);
            last_kf_corrected_center_.x = corrected.at<float>(0);
            last_kf_corrected_center_.y = corrected.at<float>(1);
        } else {
            kf_.statePost = kf_.statePre;
            kf_.errorCovPost = kf_.errorCovPre;
            last_kf_corrected_center_ = kf_pred_center;
        }
        last_measurement_accepted_ = kf_update;
    } else {
        last_measurement_accepted_ = model_update;
        last_kf_r_ = 0.0f;
        last_kf_innov_ = 0.0f;
        last_kf_innov_thresh_ = 0.0f;
    }

    object_center_ = measured_center;
    object_center_.x = std::min(std::max(object_center_.x, 0.0f),
        static_cast<float>(image_size_.width - 1));
    object_center_.y = std::min(std::max(object_center_.y, 0.0f),
        static_cast<float>(image_size_.height - 1));
    if (params_.use_kf && kf_initialized_) {
        const cv::Mat &cov = kf_.errorCovPost;
        last_kf_trace_ = cov.at<float>(0, 0) + cov.at<float>(1, 1);
        UpdateWindow(kf_trace_history_, last_kf_trace_, params_.kf_trace_window);
        float p_min = WindowMin(kf_trace_history_, last_kf_trace_);
        float p_max = WindowMax(kf_trace_history_, last_kf_trace_);
        last_kf_uncert_ = Clamp01((last_kf_trace_ - p_min) /
            (p_max - p_min + params_.apce_norm_eps));
    } else {
        last_kf_trace_ = 0.0f;
        last_kf_uncert_ = 0.0f;
    }

    float s_min = params_.search_scale_min;
    float s_max = params_.search_scale_max;
    if (s_min > s_max) {
        std::swap(s_min, s_max);
    }
    search_scale_factor_ = s_min + (s_max - s_min) *
        std::max(1.0f - psr_conf, last_kf_uncert_);

    if (model_update) {
        float new_scale = dsst_.GetScale(frame, object_center_);
        if (std::isfinite(new_scale) && new_scale > 0.0f) {
            current_scale_factor_ = new_scale;
        }
    }
    const float prev_w = bounding_box_.width > 0.0f ? bounding_box_.width : original_target_size_.width;
    const float prev_h = bounding_box_.height > 0.0f ? bounding_box_.height : original_target_size_.height;
    bounding_box_.x = object_center_.x - current_scale_factor_ * original_target_size_.width / 2.0f;
    bounding_box_.y = object_center_.y - current_scale_factor_ * original_target_size_.height / 2.0f;
    bounding_box_.width = current_scale_factor_ * original_target_size_.width;
    bounding_box_.height = current_scale_factor_ * original_target_size_.height;
    float size_conf = Clamp01(psr_conf * (1.0f - last_kf_uncert_));
    if (!model_update) {
        size_conf = 0.0f;
    }
    const float smooth_w = (1.0f - size_conf) * prev_w + size_conf * bounding_box_.width;
    const float smooth_h = (1.0f - size_conf) * prev_h + size_conf * bounding_box_.height;
    bounding_box_.width = smooth_w;
    bounding_box_.height = smooth_h;
    bounding_box_.x = object_center_.x - smooth_w / 2.0f;
    bounding_box_.y = object_center_.y - smooth_h / 2.0f;


    if (lr_scale > 0.0f) {
        if (params_.use_segmentation) {
            cv::Mat hsv_image = BgrToHsv(frame);
            UpdateHistograms(hsv_image, bounding_box_, lr_scale);
            filter_mask_ = SegmentRegion(hsv_image, object_center_, template_size_,
                original_target_size_, current_scale_factor_);
            if (filter_mask_.empty()) {
                filter_mask_ = default_mask_;
            } else {
                cv::resize(filter_mask_, filter_mask_, yf_.size(), 0, 0, cv::INTER_NEAREST);
                if (CheckMaskArea(filter_mask_, default_mask_area_)) {
                    cv::dilate(filter_mask_, filter_mask_, erode_element_);
                } else {
                    filter_mask_ = default_mask_;
                }
            }
        } else {
            filter_mask_ = default_mask_;
        }

        UpdateCsrFilter(frame, filter_mask_, lr_scale);
        dsst_.Update(frame, object_center_);
    }
    bounding_box = bounding_box_;
    return true;
}

bool CsrtTracker::Init(const cv::Mat &image, const cv::Rect &bounding_box) {
    cv::Mat frame;
    if (image.channels() == 1) {
        cv::cvtColor(image, frame, cv::COLOR_GRAY2BGR);
    } else {
        frame = image;
    }

    current_scale_factor_ = 1.0f;
    image_size_ = frame.size();
    bounding_box_ = bounding_box;
    cell_size_ = cvFloor(std::min(4.0, std::max(1.0, static_cast<double>(
        cvCeil((bounding_box.width * bounding_box.height) / 400.0)))));
    original_target_size_ = bounding_box.size();

    template_size_.width = static_cast<float>(cvFloor(original_target_size_.width + params_.padding *
        std::sqrt(original_target_size_.width * original_target_size_.height)));
    template_size_.height = static_cast<float>(cvFloor(original_target_size_.height + params_.padding *
        std::sqrt(original_target_size_.width * original_target_size_.height)));
    template_size_.width = template_size_.height =
        (template_size_.width + template_size_.height) / 2.0f;

    rescale_ratio_ = std::sqrt((params_.template_size * params_.template_size) /
        (template_size_.width * template_size_.height));
    if (rescale_ratio_ > 1.0f) {
        rescale_ratio_ = 1.0f;
    }

    rescaled_template_size_ = cv::Size2i(cvFloor(template_size_.width * rescale_ratio_),
        cvFloor(template_size_.height * rescale_ratio_));

    object_center_ = cv::Point2f(
        static_cast<float>(bounding_box.x) + original_target_size_.width / 2.0f,
        static_cast<float>(bounding_box.y) + original_target_size_.height / 2.0f);

    apce_history_.clear();
    kf_trace_history_.clear();
    last_apce_ = 0.0f;
    last_apce_norm_ = 0.0f;
    last_apce_mean_ = 0.0f;
    init_apce_ = 0.0f;
    last_apce_valid_ = false;
    last_kf_trace_ = 0.0f;
    last_kf_uncert_ = 0.0f;
    last_measurement_accepted_ = true;
    search_scale_factor_ = 1.0f;

    if (params_.use_kf) {
        kf_ = cv::KalmanFilter(4, 2, 0, CV_32F);
        kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);
        kf_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f);
        kf_.processNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
        kf_.processNoiseCov.at<float>(0, 0) = params_.kf_q_pos;
        kf_.processNoiseCov.at<float>(1, 1) = params_.kf_q_pos;
        kf_.processNoiseCov.at<float>(2, 2) = params_.kf_q_vel;
        kf_.processNoiseCov.at<float>(3, 3) = params_.kf_q_vel;
        kf_.measurementNoiseCov = (cv::Mat_<float>(2, 2) <<
            params_.kf_r_min, 0.0f,
            0.0f, params_.kf_r_min);
        kf_.errorCovPost = cv::Mat::eye(4, 4, CV_32F) * params_.kf_p_init;
        kf_.statePost = (cv::Mat_<float>(4, 1) <<
            object_center_.x, object_center_.y, 0.0f, 0.0f);
        kf_initialized_ = true;
    } else {
        kf_initialized_ = false;
    }

    yf_ = GaussianShapedLabels(params_.gsl_sigma,
        rescaled_template_size_.width / cell_size_,
        rescaled_template_size_.height / cell_size_);

    if (params_.window_function == "hann") {
        window_ = GetHannWindow(cv::Size(yf_.cols, yf_.rows));
    } else if (params_.window_function == "cheb") {
        window_ = GetChebyshevWindow(cv::Size(yf_.cols, yf_.rows), params_.cheb_attenuation);
    } else if (params_.window_function == "kaiser") {
        window_ = GetKaiserWindow(cv::Size(yf_.cols, yf_.rows), params_.kaiser_alpha);
    } else {
        return false;
    }

    cv::Size2i scaled_obj_size(cvFloor(original_target_size_.width * rescale_ratio_ / cell_size_),
        cvFloor(original_target_size_.height * rescale_ratio_ / cell_size_));
    int x0 = std::max((yf_.size().width - scaled_obj_size.width) / 2 - 1, 0);
    int y0 = std::max((yf_.size().height - scaled_obj_size.height) / 2 - 1, 0);
    default_mask_ = cv::Mat::zeros(yf_.size(), CV_32FC1);
    default_mask_(cv::Rect(x0, y0, scaled_obj_size.width, scaled_obj_size.height)) = 1.0f;
    default_mask_area_ = static_cast<float>(cv::sum(default_mask_)[0]);

    if (params_.use_segmentation) {
        cv::Mat hsv_img = BgrToHsv(frame);
        hist_foreground_ = Histogram(hsv_img.channels(), params_.histogram_bins);
        hist_background_ = Histogram(hsv_img.channels(), params_.histogram_bins);
        ExtractHistograms(hsv_img, bounding_box_, hist_foreground_, hist_background_);
        filter_mask_ = SegmentRegion(hsv_img, object_center_, template_size_,
            original_target_size_, current_scale_factor_);

        if (filter_mask_.empty()) {
            filter_mask_ = default_mask_;
        } else {
            if (!preset_mask_.empty()) {
                cv::Mat padded_mask = cv::Mat::zeros(filter_mask_.size(), filter_mask_.type());
                int sx = std::max(static_cast<int>(cvFloor(padded_mask.cols / 2.0 - preset_mask_.cols / 2.0)) - 1, 0);
                int sy = std::max(static_cast<int>(cvFloor(padded_mask.rows / 2.0 - preset_mask_.rows / 2.0)) - 1, 0);
                preset_mask_.copyTo(padded_mask(cv::Rect(sx, sy, preset_mask_.cols, preset_mask_.rows)));
                filter_mask_ = filter_mask_.mul(padded_mask);
            }

            erode_element_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
            cv::resize(filter_mask_, filter_mask_, yf_.size(), 0, 0, cv::INTER_NEAREST);
            if (CheckMaskArea(filter_mask_, default_mask_area_)) {
                cv::dilate(filter_mask_, filter_mask_, erode_element_);
            } else {
                filter_mask_ = default_mask_;
            }
        }
    } else {
        filter_mask_ = default_mask_;
    }

    cv::Mat patch = GetSubwindow(frame, object_center_,
        cvFloor(current_scale_factor_ * template_size_.width),
        cvFloor(current_scale_factor_ * template_size_.height));
    cv::resize(patch, patch, rescaled_template_size_, 0, 0, cv::INTER_CUBIC);
    std::vector<cv::Mat> patch_features = GetFeatures(patch, yf_.size());
    std::vector<cv::Mat> fft_features = FourierTransformFeatures(patch_features);

    csr_filter_ = CreateCsrFilter(fft_features, yf_, filter_mask_);

    if (params_.use_channel_weights) {
        cv::Mat current_resp;
        filter_weights_.assign(csr_filter_.size(), 0.0f);
        float sum_weights = 0.0f;
        for (size_t i = 0; i < csr_filter_.size(); ++i) {
            cv::mulSpectrums(fft_features[i], csr_filter_[i], current_resp, 0, true);
            cv::idft(current_resp, current_resp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
            double max_val = 0.0;
            cv::minMaxLoc(current_resp, nullptr, &max_val, nullptr, nullptr);
            sum_weights += static_cast<float>(max_val);
            filter_weights_[i] = static_cast<float>(max_val);
        }
        for (size_t i = 0; i < filter_weights_.size(); ++i) {
            filter_weights_[i] /= sum_weights;
        }
    }

    dsst_ = DSST(frame, bounding_box_, template_size_, params_.number_of_scales,
        params_.scale_step, params_.scale_model_max_area, params_.scale_sigma_factor,
        params_.scale_lr);

    return true;
}

}  // namespace csrt
