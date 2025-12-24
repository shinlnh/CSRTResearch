#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace csrt {

class DSST {
public:
    DSST() = default;
    DSST(const cv::Mat &image, const cv::Rect2f &bounding_box, const cv::Size2f &template_size,
        int number_of_scales, float scale_step, float max_model_area,
        float sigma_factor, float scale_learn_rate);

    void Update(const cv::Mat &image, const cv::Point2f &object_center);
    float GetScale(const cv::Mat &image, const cv::Point2f &object_center);

private:
    cv::Mat GetScaleFeatures(const cv::Mat &image, const cv::Point2f &pos,
        const cv::Size2f &base_target_size, float current_scale,
        const std::vector<float> &scale_factors, const cv::Mat &scale_window,
        const cv::Size &scale_model_size);

    cv::Size scale_model_size_;
    cv::Mat ys_;
    cv::Mat ysf_;
    cv::Mat scale_window_;
    std::vector<float> scale_factors_;
    cv::Mat sf_num_;
    cv::Mat sf_den_;
    float scale_sigma_ = 0.0f;
    float min_scale_factor_ = 0.0f;
    float max_scale_factor_ = 0.0f;
    float current_scale_factor_ = 1.0f;
    int scales_count_ = 0;
    float scale_step_ = 0.0f;
    float max_model_area_ = 0.0f;
    float sigma_factor_ = 0.0f;
    float learn_rate_ = 0.0f;
    cv::Size original_target_size_;
};

}  // namespace csrt
