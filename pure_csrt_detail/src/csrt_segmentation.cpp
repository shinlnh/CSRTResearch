#include "../inc/csrt_segmentation.hpp"

#include <cmath>
#include <limits>

namespace csrt {

Histogram::Histogram(int num_dimensions, int num_bins_per_dimension)
    : num_bins_per_dim_(num_bins_per_dimension), num_dims_(num_dimensions) {
    size_ = cvFloor(std::pow(num_bins_per_dim_, num_dims_));
    bins_.assign(size_, 0.0);
    dim_id_coef_.assign(num_dims_, 1);
    for (int i = 0; i < num_dims_ - 1; ++i) {
        dim_id_coef_[i] = static_cast<int>(std::pow(num_bins_per_dim_, num_dims_ - 1 - i));
    }
}

void Histogram::ExtractForegroundHistogram(std::vector<cv::Mat> &img_channels,
    cv::Mat weights, bool use_mat_weights, int x1, int y1, int x2, int y2) {
    if (!use_mat_weights) {
        double cx = x1 + (x2 - x1) / 2.0;
        double cy = y1 + (y2 - y1) / 2.0;
        double kernel_size_width = 1.0 / (0.5 * static_cast<double>(x2 - x1) * 1.4142 + 1.0);
        double kernel_size_height = 1.0 / (0.5 * static_cast<double>(y2 - y1) * 1.4142 + 1.0);

        cv::Mat kernel_weight(img_channels[0].rows, img_channels[0].cols, CV_64FC1);
        for (int y = y1; y < y2 + 1; ++y) {
            double *weight_ptr = kernel_weight.ptr<double>(y);
            double tmp_y = std::pow((cy - y) * kernel_size_height, 2);
            for (int x = x1; x < x2 + 1; ++x) {
                weight_ptr[x] = KernelProfileEpanechnikov(std::pow((cx - x) * kernel_size_width, 2) + tmp_y);
            }
        }
        weights = kernel_weight;
    }

    double range_per_bin_inverse = static_cast<double>(num_bins_per_dim_) / 256.0;
    double sum = 0.0;
    for (int y = y1; y < y2 + 1; ++y) {
        std::vector<const uchar *> data_ptr(num_dims_);
        for (int dim = 0; dim < num_dims_; ++dim) {
            data_ptr[dim] = img_channels[dim].ptr<uchar>(y);
        }
        const double *weight_ptr = weights.ptr<double>(y);

        for (int x = x1; x < x2 + 1; ++x) {
            int id = 0;
            for (int dim = 0; dim < num_dims_; ++dim) {
                id += dim_id_coef_[dim] * cvFloor(range_per_bin_inverse * data_ptr[dim][x]);
            }
            bins_[id] += weight_ptr[x];
            sum += weight_ptr[x];
        }
    }

    sum = 1.0 / sum;
    for (int i = 0; i < size_; ++i) {
        bins_[i] *= sum;
    }
}

void Histogram::ExtractBackgroundHistogram(std::vector<cv::Mat> &img_channels,
    int x1, int y1, int x2, int y2,
    int outer_x1, int outer_y1, int outer_x2, int outer_y2) {
    double range_per_bin_inverse = static_cast<double>(num_bins_per_dim_) / 256.0;
    double sum = 0.0;

    for (int y = outer_y1; y < outer_y2; ++y) {
        std::vector<const uchar *> data_ptr(num_dims_);
        for (int dim = 0; dim < num_dims_; ++dim) {
            data_ptr[dim] = img_channels[dim].ptr<uchar>(y);
        }

        for (int x = outer_x1; x < outer_x2; ++x) {
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                continue;
            }

            int id = 0;
            for (int dim = 0; dim < num_dims_; ++dim) {
                id += dim_id_coef_[dim] * cvFloor(range_per_bin_inverse * data_ptr[dim][x]);
            }
            bins_[id] += 1.0;
            sum += 1.0;
        }
    }

    sum = 1.0 / sum;
    for (int i = 0; i < size_; ++i) {
        bins_[i] *= sum;
    }
}

cv::Mat Histogram::BackProject(std::vector<cv::Mat> &img_channels) {
    cv::Mat back_project(img_channels[0].rows, img_channels[0].cols, CV_64FC1);
    double range_per_bin_inverse = static_cast<double>(num_bins_per_dim_) / 256.0;

    for (int y = 0; y < img_channels[0].rows; ++y) {
        double *back_ptr = back_project.ptr<double>(y);
        std::vector<const uchar *> data_ptr(num_dims_);
        for (int dim = 0; dim < num_dims_; ++dim) {
            data_ptr[dim] = img_channels[dim].ptr<uchar>(y);
        }

        for (int x = 0; x < img_channels[0].cols; ++x) {
            int id = 0;
            for (int dim = 0; dim < num_dims_; ++dim) {
                id += dim_id_coef_[dim] * cvFloor(range_per_bin_inverse * data_ptr[dim][x]);
            }
            back_ptr[x] = bins_[id];
        }
    }
    return back_project;
}

std::vector<double> Histogram::GetHistogramVector() const {
    return bins_;
}

void Histogram::SetHistogramVector(const double *values) {
    for (size_t i = 0; i < bins_.size(); ++i) {
        bins_[i] = values[i];
    }
}

std::pair<cv::Mat, cv::Mat> Segment::ComputePosteriors(
    std::vector<cv::Mat> &img_channels, int x1, int y1, int x2, int y2,
    cv::Mat weights, cv::Mat fg_prior, cv::Mat bg_prior,
    const Histogram &fg_hist_prior, int num_bins_per_channel) {
    CV_Assert(!img_channels.empty());

    x1 = std::min(std::max(x1, 0), img_channels[0].cols - 1);
    y1 = std::min(std::max(y1, 0), img_channels[0].rows - 1);
    x2 = std::max(std::min(x2, img_channels[0].cols - 1), 0);
    y2 = std::max(std::min(y2, img_channels[0].rows - 1), 0);

    int offset_x = (x2 - x1) / 3;
    int offset_y = (y2 - y1) / 3;
    int outer_y1 = std::max(0, y1 - offset_y);
    int outer_y2 = std::min(img_channels[0].rows, y2 + offset_y + 1);
    int outer_x1 = std::max(0, x1 - offset_x);
    int outer_x2 = std::min(img_channels[0].cols, x2 + offset_x + 1);

    Histogram hist_target =
        (fg_hist_prior.NumBinsPerDim() == num_bins_per_channel && fg_hist_prior.NumDims() == static_cast<int>(img_channels.size()))
        ? fg_hist_prior
        : Histogram(static_cast<int>(img_channels.size()), num_bins_per_channel);
    Histogram hist_background(static_cast<int>(img_channels.size()), num_bins_per_channel);

    if (weights.cols == 0) {
        hist_target.ExtractForegroundHistogram(img_channels, cv::Mat(), false, x1, y1, x2, y2);
    } else {
        hist_target.ExtractForegroundHistogram(img_channels, weights, true, x1, y1, x2, y2);
    }

    hist_background.ExtractBackgroundHistogram(img_channels, x1, y1, x2, y2,
        outer_x1, outer_y1, outer_x2, outer_y2);

    double factor = std::sqrt(1000.0 / ((x2 - x1) * (y2 - y1)));
    if (factor > 1.0) {
        factor = 1.0;
    }

    cv::Size new_size(cvFloor((x2 - x1) * factor), cvFloor((y2 - y1) * factor));

    cv::Rect roi_inner = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    std::vector<cv::Mat> roi_channels(img_channels.size());
    for (size_t i = 0; i < img_channels.size(); ++i) {
        cv::resize(img_channels[i](roi_inner), roi_channels[i], new_size);
    }

    cv::Mat fg_prior_scaled;
    if (fg_prior.cols == 0) {
        fg_prior_scaled = 0.5 * cv::Mat::ones(new_size, CV_64FC1);
    } else {
        cv::resize(fg_prior(roi_inner), fg_prior_scaled, new_size);
    }

    cv::Mat bg_prior_scaled;
    if (bg_prior.cols == 0) {
        bg_prior_scaled = 0.5 * cv::Mat::ones(new_size, CV_64FC1);
    } else {
        cv::resize(bg_prior(roi_inner), bg_prior_scaled, new_size);
    }

    cv::Mat foreground_likelihood = hist_target.BackProject(roi_channels).mul(fg_prior_scaled);
    cv::Mat background_likelihood = hist_background.BackProject(roi_channels).mul(bg_prior_scaled);

    double p_b = std::sqrt((std::pow(outer_x2 - outer_x1, 2) + std::pow(outer_y2 - outer_y1, 2)) /
        (std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2)));
    double p_o = 1.0 / (p_b + 1.0);

    cv::Mat prob_o(new_size, foreground_likelihood.type());
    prob_o = p_o * foreground_likelihood / (p_o * foreground_likelihood + p_b * background_likelihood);
    cv::Mat prob_b = 1.0 - prob_o;

    auto sized_probs = GetRegularizedSegmentation(prob_o, prob_b, fg_prior_scaled, bg_prior_scaled);

    std::pair<cv::Mat, cv::Mat> probs;
    cv::resize(sized_probs.first, probs.first, cv::Size(roi_inner.width, roi_inner.height));
    cv::resize(sized_probs.second, probs.second, cv::Size(roi_inner.width, roi_inner.height));
    return probs;
}

std::pair<cv::Mat, cv::Mat> Segment::ComputePosteriors2(
    std::vector<cv::Mat> &img_channels, int x1, int y1, int x2, int y2, double p_b,
    cv::Mat fg_prior, cv::Mat bg_prior, Histogram hist_target, Histogram hist_background) {
    CV_Assert(!img_channels.empty());

    x1 = std::min(std::max(x1, 0), img_channels[0].cols - 1);
    y1 = std::min(std::max(y1, 0), img_channels[0].rows - 1);
    x2 = std::max(std::min(x2, img_channels[0].cols - 1), 0);
    y2 = std::max(std::min(y2, img_channels[0].rows - 1), 0);

    int width = x2 - x1 + 1;
    int height = y2 - y1 + 1;
    width = std::min(std::max(width, 1), img_channels[0].cols);
    height = std::min(std::max(height, 1), img_channels[0].rows);

    double p_o = 1.0 - p_b;

    double factor = std::sqrt(1000.0 / (width * height));
    if (factor > 1.0) {
        factor = 1.0;
    }
    cv::Size new_size(cvFloor(width * factor), cvFloor(height * factor));

    cv::Rect roi_inner = cv::Rect(x1, y1, width, height);
    std::vector<cv::Mat> roi_channels(img_channels.size());
    for (size_t i = 0; i < img_channels.size(); ++i) {
        cv::resize(img_channels[i](roi_inner), roi_channels[i], new_size);
    }

    cv::Mat fg_prior_scaled;
    if (fg_prior.cols == 0) {
        fg_prior_scaled = 0.5 * cv::Mat::ones(new_size, CV_64FC1);
    } else {
        cv::resize(fg_prior(roi_inner), fg_prior_scaled, new_size);
    }

    cv::Mat bg_prior_scaled;
    if (bg_prior.cols == 0) {
        bg_prior_scaled = 0.5 * cv::Mat::ones(new_size, CV_64FC1);
    } else {
        cv::resize(bg_prior(roi_inner), bg_prior_scaled, new_size);
    }

    cv::Mat foreground_likelihood = hist_target.BackProject(roi_channels).mul(fg_prior_scaled);
    cv::Mat background_likelihood = hist_background.BackProject(roi_channels).mul(bg_prior_scaled);

    cv::Mat prob_o(new_size, foreground_likelihood.type());
    prob_o = p_o * foreground_likelihood / (p_o * foreground_likelihood + p_b * background_likelihood);
    cv::Mat prob_b = 1.0 - prob_o;

    auto sized_probs = GetRegularizedSegmentation(prob_o, prob_b, fg_prior_scaled, bg_prior_scaled);

    std::pair<cv::Mat, cv::Mat> probs;
    cv::resize(sized_probs.first, probs.first, cv::Size(roi_inner.width, roi_inner.height));
    cv::resize(sized_probs.second, probs.second, cv::Size(roi_inner.width, roi_inner.height));
    return probs;
}

std::pair<cv::Mat, cv::Mat> Segment::ComputePosteriors2(
    std::vector<cv::Mat> &img_channels, cv::Mat fg_prior, cv::Mat bg_prior,
    Histogram hist_target, Histogram hist_background) {
    CV_Assert(!img_channels.empty());

    int x1 = 0;
    int y1 = 0;
    int x2 = img_channels[0].cols - 1;
    int y2 = img_channels[0].rows - 1;

    double factor = std::sqrt(1000.0 / ((x2 - x1) * (y2 - y1)));
    if (factor > 1.0) {
        factor = 1.0;
    }

    cv::Size new_size(cvFloor((x2 - x1) * factor), cvFloor((y2 - y1) * factor));
    cv::Rect roi_inner = cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

    std::vector<cv::Mat> roi_channels(img_channels.size());
    for (size_t i = 0; i < img_channels.size(); ++i) {
        cv::resize(img_channels[i](roi_inner), roi_channels[i], new_size);
    }

    cv::Mat fg_prior_scaled;
    if (fg_prior.cols == 0) {
        fg_prior_scaled = 0.5 * cv::Mat::ones(new_size, CV_64FC1);
    } else {
        cv::resize(fg_prior(roi_inner), fg_prior_scaled, new_size);
    }

    cv::Mat bg_prior_scaled;
    if (bg_prior.cols == 0) {
        bg_prior_scaled = 0.5 * cv::Mat::ones(new_size, CV_64FC1);
    } else {
        cv::resize(bg_prior(roi_inner), bg_prior_scaled, new_size);
    }

    cv::Mat foreground_likelihood = hist_target.BackProject(roi_channels).mul(fg_prior_scaled);
    cv::Mat background_likelihood = hist_background.BackProject(roi_channels).mul(bg_prior_scaled);

    double p_b = 5.0 / 3.0;
    double p_o = 1.0 / (p_b + 1.0);

    cv::Mat prob_o(new_size, foreground_likelihood.type());
    prob_o = p_o * foreground_likelihood / (p_o * foreground_likelihood + p_b * background_likelihood);
    cv::Mat prob_b = 1.0 - prob_o;

    auto sized_probs = GetRegularizedSegmentation(prob_o, prob_b, fg_prior_scaled, bg_prior_scaled);

    std::pair<cv::Mat, cv::Mat> probs;
    cv::resize(sized_probs.first, probs.first, cv::Size(roi_inner.width, roi_inner.height));
    cv::resize(sized_probs.second, probs.second, cv::Size(roi_inner.width, roi_inner.height));
    return probs;
}

std::pair<cv::Mat, cv::Mat> Segment::GetRegularizedSegmentation(
    cv::Mat &prob_o, cv::Mat &prob_b, cv::Mat &prior_o, cv::Mat &prior_b) {
    int hsize = cvFloor(std::max(1.0, static_cast<double>(cvFloor(prob_b.cols * 3.0 / 50.0 + 0.5))));
    int lambda_size = hsize * 2 + 1;

    cv::Mat lambda(lambda_size, lambda_size, CV_64FC1);
    double std2 = std::pow(hsize / 3.0, 2);
    double sum_lambda = 0.0;
    for (int y = -hsize; y < hsize + 1; ++y) {
        double *lambda_ptr = lambda.ptr<double>(y + hsize);
        double tmp_y = y * y;
        for (int x = -hsize; x < hsize + 1; ++x) {
            double tmp_gauss = Gaussian(x * x, tmp_y, std2);
            lambda_ptr[x + hsize] = tmp_gauss;
            sum_lambda += tmp_gauss;
        }
    }
    sum_lambda -= lambda.at<double>(hsize, hsize);
    lambda.at<double>(hsize, hsize) = 0.0;
    sum_lambda = 1.0 / sum_lambda;
    lambda = lambda * sum_lambda;

    cv::Mat lambda2 = lambda.clone();
    lambda2.at<double>(hsize, hsize) = 1.0;

    double terminate_thr = 1e-1;
    double log_like = std::numeric_limits<double>::max();
    int max_iter = 50;

    cv::Mat qsum_o(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat qsum_b(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat si_o(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat si_b(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat ssum_o(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat ssum_b(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat qi_o(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat qi_b(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat log_qo(prior_o.rows, prior_o.cols, prior_o.type());
    cv::Mat log_qb(prior_o.rows, prior_o.cols, prior_o.type());

    for (int i = 0; i < max_iter; ++i) {
        cv::Mat p_io = prior_o.mul(prob_o) + std::numeric_limits<double>::epsilon();
        cv::Mat p_ib = prior_b.mul(prob_b) + std::numeric_limits<double>::epsilon();

        cv::filter2D(prior_o, si_o, -1, lambda, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(prior_b, si_b, -1, lambda, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        si_o = si_o.mul(prior_o);
        si_b = si_b.mul(prior_b);
        cv::Mat norm_si = 1.0 / (si_o + si_b);
        si_o = si_o.mul(norm_si);
        si_b = si_b.mul(norm_si);
        cv::filter2D(si_o, ssum_o, -1, lambda2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(si_b, ssum_b, -1, lambda2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        cv::filter2D(p_io, qi_o, -1, lambda, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(p_ib, qi_b, -1, lambda, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        qi_o = qi_o.mul(p_io);
        qi_b = qi_b.mul(p_ib);
        cv::Mat norm_qi = 1.0 / (qi_o + qi_b);
        qi_o = qi_o.mul(norm_qi);
        qi_b = qi_b.mul(norm_qi);
        cv::filter2D(qi_o, qsum_o, -1, lambda2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(qi_b, qsum_b, -1, lambda2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        prior_o = (qsum_o + ssum_o) * 0.25;
        prior_b = (qsum_b + ssum_b) * 0.25;
        cv::Mat norm_pi = 1.0 / (prior_o + prior_b);
        prior_o = prior_o.mul(norm_pi);
        prior_b = prior_b.mul(norm_pi);

        cv::log(qsum_o, log_qo);
        cv::log(qsum_b, log_qb);
        cv::Scalar mean = cv::sum(log_qo + log_qb);
        double log_like_new = -mean.val[0] / (2.0 * qsum_o.rows * qsum_o.cols);
        if (std::abs(log_like - log_like_new) < terminate_thr) {
            break;
        }
        log_like = log_like_new;
    }

    return std::make_pair(qsum_o, qsum_b);
}

}  // namespace csrt
