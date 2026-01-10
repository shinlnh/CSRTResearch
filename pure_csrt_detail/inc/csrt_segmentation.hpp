#pragma once

#include <opencv2/opencv.hpp>

#include <utility>
#include <vector>

namespace csrt {

class Histogram {
public:
    Histogram() : num_bins_per_dim_(0), num_dims_(0) {}
    Histogram(int num_dimensions, int num_bins_per_dimension = 8);

    void ExtractForegroundHistogram(std::vector<cv::Mat> &img_channels, cv::Mat weights,
        bool use_mat_weights, int x1, int y1, int x2, int y2);
    void ExtractBackgroundHistogram(std::vector<cv::Mat> &img_channels,
        int x1, int y1, int x2, int y2,
        int outer_x1, int outer_y1, int outer_x2, int outer_y2);
    cv::Mat BackProject(std::vector<cv::Mat> &img_channels);

    std::vector<double> GetHistogramVector() const;
    void SetHistogramVector(const double *values);

    int NumBinsPerDim() const { return num_bins_per_dim_; }
    int NumDims() const { return num_dims_; }

private:
    int num_bins_per_dim_;
    int num_dims_;
    int size_;
    std::vector<double> bins_;
    std::vector<int> dim_id_coef_;

    double KernelProfileEpanechnikov(double x) const { return (x <= 1.0) ? (2.0 / CV_PI) * (1.0 - x) : 0.0; }
};

class Segment {
public:
    static std::pair<cv::Mat, cv::Mat> ComputePosteriors(
        std::vector<cv::Mat> &img_channels, int x1, int y1, int x2, int y2,
        cv::Mat weights, cv::Mat fg_prior, cv::Mat bg_prior,
        const Histogram &fg_hist_prior, int num_bins_per_channel = 16);

    static std::pair<cv::Mat, cv::Mat> ComputePosteriors2(
        std::vector<cv::Mat> &img_channels, int x1, int y1, int x2, int y2, double p_b,
        cv::Mat fg_prior, cv::Mat bg_prior, Histogram hist_target, Histogram hist_background);

    static std::pair<cv::Mat, cv::Mat> ComputePosteriors2(
        std::vector<cv::Mat> &img_channels, cv::Mat fg_prior, cv::Mat bg_prior,
        Histogram hist_target, Histogram hist_background);

private:
    static std::pair<cv::Mat, cv::Mat> GetRegularizedSegmentation(
        cv::Mat &prob_o, cv::Mat &prob_b, cv::Mat &prior_o, cv::Mat &prior_b);

    static double Gaussian(double x2, double y2, double std2) {
        return std::exp(-(x2 + y2) / (2.0 * std2)) / (2.0 * CV_PI * std2);
    }
};

}  // namespace csrt
