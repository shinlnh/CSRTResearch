#include "../inc/csrt_scale.hpp"

#include "../inc/csrt_utils.hpp"

namespace csrt {

class ParallelGetScaleFeatures : public cv::ParallelLoopBody {
public:
    ParallelGetScaleFeatures(const cv::Mat &image, const cv::Point2f &pos,
        const cv::Size2f &base_target_size, float current_scale,
        const std::vector<float> &scale_factors, const cv::Mat &scale_window,
        const cv::Size &scale_model_size, int col_len, cv::Mat &result)
        : image_(image), pos_(pos), base_target_size_(base_target_size),
          current_scale_(current_scale), scale_factors_(scale_factors),
          scale_window_(scale_window), scale_model_size_(scale_model_size),
          col_len_(col_len), result_(result) {}

    void operator()(const cv::Range &range) const override {
        for (int s = range.start; s < range.end; ++s) {
            cv::Size patch_size(
                static_cast<int>(current_scale_ * scale_factors_[s] * base_target_size_.width),
                static_cast<int>(current_scale_ * scale_factors_[s] * base_target_size_.height));
            cv::Mat img_patch = GetSubwindow(image_, pos_, patch_size.width, patch_size.height);
            img_patch.convertTo(img_patch, CV_32FC3);
            cv::resize(img_patch, img_patch, scale_model_size_, 0, 0, cv::INTER_LINEAR);
            std::vector<cv::Mat> hog = GetFeaturesHog(img_patch, 4);
            for (size_t i = 0; i < hog.size(); ++i) {
                hog[i] = hog[i].t();
                hog[i] = scale_window_.at<float>(0, s) * hog[i].reshape(0, col_len_);
                hog[i].copyTo(result_(cv::Rect(cv::Point(s, static_cast<int>(i) * col_len_), hog[i].size())));
            }
        }
    }

private:
    cv::Mat image_;
    cv::Point2f pos_;
    cv::Size2f base_target_size_;
    float current_scale_;
    std::vector<float> scale_factors_;
    cv::Mat scale_window_;
    cv::Size scale_model_size_;
    int col_len_;
    cv::Mat result_;
};

DSST::DSST(const cv::Mat &image, const cv::Rect2f &bounding_box, const cv::Size2f &template_size,
    int number_of_scales, float scale_step, float max_model_area,
    float sigma_factor, float scale_learn_rate)
    : scales_count_(number_of_scales), scale_step_(scale_step),
      max_model_area_(max_model_area), sigma_factor_(sigma_factor),
      learn_rate_(scale_learn_rate) {
    original_target_size_ = bounding_box.size();
    cv::Point2f object_center(bounding_box.x + original_target_size_.width / 2.0f,
        bounding_box.y + original_target_size_.height / 2.0f);

    current_scale_factor_ = 1.0f;
    if (scales_count_ % 2 == 0) {
        scales_count_++;
    }

    scale_sigma_ = static_cast<float>(std::sqrt(scales_count_) * sigma_factor_);

    min_scale_factor_ = static_cast<float>(std::pow(scale_step_,
        cvCeil(std::log(std::max(5.0 / template_size.width, 5.0 / template_size.height)) / std::log(scale_step_))));
    max_scale_factor_ = static_cast<float>(std::pow(scale_step_,
        cvFloor(std::log(std::min(static_cast<float>(image.rows) / bounding_box.width,
        static_cast<float>(image.cols) / bounding_box.height)) / std::log(scale_step_))));

    ys_ = cv::Mat(1, scales_count_, CV_32FC1);
    for (int i = 0; i < ys_.cols; ++i) {
        float ss = static_cast<float>(i + 1) - cvCeil(scales_count_ / 2.0f);
        ys_.at<float>(0, i) = static_cast<float>(std::exp(-0.5 * std::pow(ss, 2) / std::pow(scale_sigma_, 2)));
        float sf = static_cast<float>(i + 1);
        scale_factors_.push_back(std::pow(scale_step_, cvCeil(scales_count_ / 2.0f) - sf));
    }

    scale_window_ = GetHannWindow(cv::Size(scales_count_, 1));

    float scale_model_factor = 1.0f;
    if (template_size.width * template_size.height * std::pow(scale_model_factor, 2) > max_model_area_) {
        scale_model_factor = std::sqrt(max_model_area_ /
            (template_size.width * template_size.height));
    }
    scale_model_size_ = cv::Size(cvFloor(template_size.width * scale_model_factor),
        cvFloor(template_size.height * scale_model_factor));

    cv::Mat scale_resp = GetScaleFeatures(image, object_center, original_target_size_,
        current_scale_factor_, scale_factors_, scale_window_, scale_model_size_);

    cv::Mat ysf_row(ys_.size(), CV_32FC2);
    cv::dft(ys_, ysf_row, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT, 0);
    ysf_ = cv::repeat(ysf_row, scale_resp.rows, 1);

    cv::Mat fscale_resp;
    cv::dft(scale_resp, fscale_resp, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(ysf_, fscale_resp, sf_num_, 0, true);

    cv::Mat sf_den_all;
    cv::mulSpectrums(fscale_resp, fscale_resp, sf_den_all, 0, true);
    cv::reduce(sf_den_all, sf_den_, 0, cv::REDUCE_SUM, -1);
}

cv::Mat DSST::GetScaleFeatures(const cv::Mat &image, const cv::Point2f &pos,
    const cv::Size2f &base_target_size, float current_scale,
    const std::vector<float> &scale_factors, const cv::Mat &scale_window,
    const cv::Size &scale_model_size) {
    cv::Mat result;
    int col_len = 0;

    cv::Size patch_size(
        cvFloor(current_scale * scale_factors[0] * base_target_size.width),
        cvFloor(current_scale * scale_factors[0] * base_target_size.height));
    cv::Mat img_patch = GetSubwindow(image, pos, patch_size.width, patch_size.height);
    img_patch.convertTo(img_patch, CV_32FC3);
    cv::resize(img_patch, img_patch, scale_model_size, 0, 0, cv::INTER_LINEAR);

    std::vector<cv::Mat> hog = GetFeaturesHog(img_patch, 4);
    result = cv::Mat(cv::Size(static_cast<int>(scale_factors.size()),
        hog[0].cols * hog[0].rows * static_cast<int>(hog.size())), CV_32F);
    col_len = hog[0].cols * hog[0].rows;

    for (size_t i = 0; i < hog.size(); ++i) {
        hog[i] = hog[i].t();
        hog[i] = scale_window.at<float>(0, 0) * hog[i].reshape(0, col_len);
        hog[i].copyTo(result(cv::Rect(cv::Point(0, static_cast<int>(i) * col_len), hog[i].size())));
    }

    ParallelGetScaleFeatures parallel_worker(image, pos, base_target_size, current_scale,
        scale_factors, scale_window, scale_model_size, col_len, result);
    cv::parallel_for_(cv::Range(1, static_cast<int>(scale_factors.size())), parallel_worker);
    return result;
}

void DSST::Update(const cv::Mat &image, const cv::Point2f &object_center) {
    cv::Mat scale_features = GetScaleFeatures(image, object_center, original_target_size_,
        current_scale_factor_, scale_factors_, scale_window_, scale_model_size_);
    cv::Mat fscale_features;
    cv::dft(scale_features, fscale_features, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);

    cv::Mat new_sf_num;
    cv::Mat new_sf_den;
    cv::Mat new_sf_den_all;
    cv::mulSpectrums(ysf_, fscale_features, new_sf_num, cv::DFT_ROWS, true);
    cv::mulSpectrums(fscale_features, fscale_features, new_sf_den_all, cv::DFT_ROWS, true);
    cv::reduce(new_sf_den_all, new_sf_den, 0, cv::REDUCE_SUM, -1);

    sf_num_ = (1.0f - learn_rate_) * sf_num_ + learn_rate_ * new_sf_num;
    sf_den_ = (1.0f - learn_rate_) * sf_den_ + learn_rate_ * new_sf_den;
}

float DSST::GetScale(const cv::Mat &image, const cv::Point2f &object_center) {
    cv::Mat scale_features = GetScaleFeatures(image, object_center, original_target_size_,
        current_scale_factor_, scale_factors_, scale_window_, scale_model_size_);

    cv::Mat fscale_features;
    cv::dft(scale_features, fscale_features, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);

    cv::mulSpectrums(fscale_features, sf_num_, fscale_features, 0, false);
    cv::Mat scale_response;
    cv::reduce(fscale_features, scale_response, 0, cv::REDUCE_SUM, -1);
    scale_response = DivideComplexMatrices(scale_response, sf_den_ + 0.01f);
    cv::idft(scale_response, scale_response, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::Point max_loc;
    cv::minMaxLoc(scale_response, nullptr, nullptr, nullptr, &max_loc);

    current_scale_factor_ *= scale_factors_[max_loc.x];
    if (current_scale_factor_ < min_scale_factor_) {
        current_scale_factor_ = min_scale_factor_;
    } else if (current_scale_factor_ > max_scale_factor_) {
        current_scale_factor_ = max_scale_factor_;
    }

    return current_scale_factor_;
}

}  // namespace csrt
