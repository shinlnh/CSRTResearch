#include "../inc/csrt_utils.hpp"

#include "../inc/csrt_color_names.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace csrt {

int Mod(int a, int b) {
    return ((a % b) + b) % b;
}

double KernelEpan(double x) {
    return (x <= 1.0) ? (2.0 / 3.14) * (1.0 - x) : 0.0;
}

cv::Mat CircShift(const cv::Mat &matrix, int dx, int dy) {
    cv::Mat output = matrix.clone();
    for (int y = 0; y < matrix.rows; ++y) {
        for (int x = 0; x < matrix.cols; ++x) {
            int new_y = Mod(y + dy + 1, matrix.rows);
            int new_x = Mod(x + dx + 1, matrix.cols);
            output.at<float>(new_y, new_x) = matrix.at<float>(y, x);
        }
    }
    return output;
}

cv::Mat GaussianShapedLabels(float sigma, int width, int height) {
    cv::Mat labels = cv::Mat::zeros(height, width, CV_32F);
    float half_w = static_cast<float>(cvFloor(width / 2));
    float half_h = static_cast<float>(cvFloor(height / 2));

    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            float val = static_cast<float>(
                std::exp((-0.5 / std::pow(sigma, 2)) *
                    (std::pow((y + 1 - half_h), 2) + std::pow((x + 1 - half_w), 2))));
            labels.at<float>(y, x) = val;
        }
    }

    labels = CircShift(labels, -cvFloor(labels.cols / 2), -cvFloor(labels.rows / 2));
    cv::Mat yf;
    cv::dft(labels, yf, cv::DFT_COMPLEX_OUTPUT);
    return yf;
}

std::vector<cv::Mat> FourierTransformFeatures(const std::vector<cv::Mat> &features) {
    std::vector<cv::Mat> out(features.size());
    cv::Mat channel;
    for (size_t i = 0; i < features.size(); ++i) {
        features[i].convertTo(channel, CV_32F);
        cv::dft(channel, channel, cv::DFT_COMPLEX_OUTPUT);
        out[i] = channel;
    }
    return out;
}

cv::Mat DivideComplexMatrices(const cv::Mat &a, const cv::Mat &b) {
    std::vector<cv::Mat> va;
    std::vector<cv::Mat> vb;
    cv::split(a, va);
    cv::split(b, vb);

    cv::Mat ar = va.at(0);
    cv::Mat ai = va.at(1);
    cv::Mat br = vb.at(0);
    cv::Mat bi = vb.at(1);

    cv::Mat denom = br.mul(br) + bi.mul(bi);
    cv::Mat real_part = (ar.mul(br) + ai.mul(bi));
    cv::Mat imag_part = (ai.mul(br) - ar.mul(bi));

    cv::divide(real_part, denom, real_part);
    cv::divide(imag_part, denom, imag_part);

    std::vector<cv::Mat> merged(2);
    merged[0] = real_part;
    merged[1] = imag_part;

    cv::Mat result;
    cv::merge(merged, result);
    return result;
}

cv::Mat GetSubwindow(const cv::Mat &image, const cv::Point2f &center, int width, int height,
    cv::Rect *valid_pixels) {
    int start_x = cvFloor(center.x) + 1 - (cvFloor(width / 2));
    int start_y = cvFloor(center.y) + 1 - (cvFloor(height / 2));

    cv::Rect roi(start_x, start_y, width, height);
    int padding_left = 0;
    int padding_right = 0;
    int padding_top = 0;
    int padding_bottom = 0;

    if (roi.x < 0) {
        padding_left = -roi.x;
        roi.x = 0;
    }
    if (roi.y < 0) {
        padding_top = -roi.y;
        roi.y = 0;
    }

    roi.width -= padding_left;
    roi.height -= padding_top;

    if (roi.x + roi.width >= image.cols) {
        padding_right = roi.x + roi.width - image.cols;
        roi.width = image.cols - roi.x;
    }
    if (roi.y + roi.height >= image.rows) {
        padding_bottom = roi.y + roi.height - image.rows;
        roi.height = image.rows - roi.y;
    }

    cv::Mat subwindow = image(roi).clone();
    cv::copyMakeBorder(subwindow, subwindow, padding_top, padding_bottom, padding_left,
        padding_right, cv::BORDER_REPLICATE);

    if (valid_pixels != nullptr) {
        *valid_pixels = cv::Rect(padding_left, padding_top, roi.width, roi.height);
    }
    return subwindow;
}

float SubpixelPeak(const cv::Mat &response, const std::string &direction, const cv::Point2f &peak) {
    int idx_center = 0;
    int idx_left = 0;
    int idx_right = 0;
    float center_val = 0.0f;
    float left_val = 0.0f;
    float right_val = 0.0f;

    if (direction == "vertical") {
        idx_center = cvRound(peak.y);
        idx_left = Mod(cvRound(peak.y) - 1, response.rows);
        idx_right = Mod(cvRound(peak.y) + 1, response.rows);
        int px = static_cast<int>(peak.x);
        center_val = response.at<float>(idx_center, px);
        left_val = response.at<float>(idx_left, px);
        right_val = response.at<float>(idx_right, px);
    } else if (direction == "horizontal") {
        idx_center = cvRound(peak.x);
        idx_left = Mod(cvRound(peak.x) - 1, response.cols);
        idx_right = Mod(cvRound(peak.x) + 1, response.cols);
        int py = static_cast<int>(peak.y);
        center_val = response.at<float>(py, idx_center);
        left_val = response.at<float>(py, idx_left);
        right_val = response.at<float>(py, idx_right);
    } else {
        return 0.0f;
    }

    float denom = 2.0f * center_val - right_val - left_val;
    if (std::abs(denom) < std::numeric_limits<float>::epsilon()) {
        return 0.0f;
    }

    float delta = 0.5f * (right_val - left_val) / denom;
    if (!std::isfinite(delta)) {
        delta = 0.0f;
    }
    return delta;
}

double GetMax(const cv::Mat &matrix) {
    double val = 0.0;
    cv::minMaxLoc(matrix, nullptr, &val, nullptr, nullptr);
    return val;
}

double GetMin(const cv::Mat &matrix) {
    double val = 0.0;
    cv::minMaxLoc(matrix, &val, nullptr, nullptr, nullptr);
    return val;
}

static float ChebPoly(int n, float x) {
    if (std::fabs(x) <= 1.0f) {
        return std::cos(n * std::acos(x));
    }
    return std::cosh(n * std::acosh(x));
}

static cv::Mat ChebWin(int n, float atten) {
    cv::Mat out(n, 1, CV_32FC1);
    float m = (n - 1) / 2.0f;
    if (n % 2 == 0) {
        m = m + 0.5f;
    }

    float tg = static_cast<float>(std::pow(10, atten / 20.0f));
    float x0 = std::cosh((1.0f / (n - 1)) * std::acosh(tg));

    float max_val = 0.0f;
    for (int nn = 0; nn < (n / 2 + 1); ++nn) {
        float sum = 0.0f;
        float val = nn - m;
        for (int i = 1; i <= m; ++i) {
            sum += ChebPoly(n - 1, x0 * static_cast<float>(std::cos(CV_PI * i / n))) *
                static_cast<float>(std::cos(2.0f * val * CV_PI * i / n));
        }
        out.at<float>(nn, 0) = tg + 2.0f * sum;
        out.at<float>(n - nn - 1, 0) = out.at<float>(nn, 0);
        if (out.at<float>(nn, 0) > max_val) {
            max_val = out.at<float>(nn, 0);
        }
    }

    for (int nn = 0; nn < n; ++nn) {
        out.at<float>(nn, 0) /= max_val;
    }

    return out;
}

static double ModifiedBessel(int order, double x) {
    const double eps = 1e-13;
    double result = 0.0;
    double m = 0.0;
    double gamma = 1.0;
    for (int i = 2; i <= order; ++i) {
        gamma *= i;
    }
    double term = std::pow(x, order) / (std::pow(2.0, order) * gamma);

    while (term > eps * result) {
        result += term;
        ++m;
        term *= (x * x) / (4.0 * m * (m + order));
    }
    return result;
}

cv::Mat GetHannWindow(cv::Size size) {
    cv::Mat hann_rows = cv::Mat::ones(size.height, 1, CV_32F);
    cv::Mat hann_cols = cv::Mat::ones(1, size.width, CV_32F);
    int nn = size.height - 1;
    if (nn != 0) {
        for (int i = 0; i < hann_rows.rows; ++i) {
            hann_rows.at<float>(i, 0) =
                static_cast<float>(0.5 * (1.0 - std::cos(2.0 * CV_PI * i / nn)));
        }
    }

    nn = size.width - 1;
    if (nn != 0) {
        for (int i = 0; i < hann_cols.cols; ++i) {
            hann_cols.at<float>(0, i) =
                static_cast<float>(0.5 * (1.0 - std::cos(2.0 * CV_PI * i / nn)));
        }
    }

    return hann_rows * hann_cols;
}

cv::Mat GetKaiserWindow(cv::Size size, float alpha) {
    cv::Mat kaiser_rows = cv::Mat::ones(size.height, 1, CV_32F);
    cv::Mat kaiser_cols = cv::Mat::ones(1, size.width, CV_32F);

    int n = size.height - 1;
    double shape = alpha;
    double den = 1.0 / ModifiedBessel(0, shape);

    for (int i = 0; i <= n; ++i) {
        double k = (2.0 * i * 1.0 / n) - 1.0;
        double x = std::sqrt(1.0 - (k * k));
        kaiser_rows.at<float>(i, 0) = static_cast<float>(ModifiedBessel(0, shape * x) * den);
    }

    n = size.width - 1;
    for (int i = 0; i <= n; ++i) {
        double k = (2.0 * i * 1.0 / n) - 1.0;
        double x = std::sqrt(1.0 - (k * k));
        kaiser_cols.at<float>(0, i) = static_cast<float>(ModifiedBessel(0, shape * x) * den);
    }

    return kaiser_rows * kaiser_cols;
}

cv::Mat GetChebyshevWindow(cv::Size size, float attenuation) {
    cv::Mat cheb_rows = ChebWin(size.height, attenuation);
    cv::Mat cheb_cols = ChebWin(size.width, attenuation).t();
    return cheb_rows * cheb_cols;
}

static void ComputeHog32D(const cv::Mat &image, cv::Mat &features, int bin_size,
    int pad_x, int pad_y) {
    const int dim_hog = 32;
    CV_Assert(pad_x >= 0);
    CV_Assert(pad_y >= 0);
    CV_Assert(image.channels() == 3);
    CV_Assert(image.depth() == CV_64F);

    const double eps = 0.0001;
    const int num_orient = 18;
    const double uu[9] = {1.000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
    const double vv[9] = {0.000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420};

    const cv::Size image_size = image.size();
    int block_w = cvFloor(static_cast<double>(image_size.width) / static_cast<double>(bin_size));
    int block_h = cvFloor(static_cast<double>(image_size.height) / static_cast<double>(bin_size));
    const cv::Size block_size(block_w, block_h);
    int out_w = std::max(block_size.width - 2, 0) + 2 * pad_x;
    int out_h = std::max(block_size.height - 2, 0) + 2 * pad_y;
    cv::Size out_size(out_w, out_h);
    const cv::Size visible = block_size * bin_size;

    cv::Mat hist = cv::Mat::zeros(cv::Size(block_size.width * num_orient, block_size.height), CV_64F);
    cv::Mat norm = cv::Mat::zeros(cv::Size(block_size.width, block_size.height), CV_64F);
    features = cv::Mat::zeros(cv::Size(out_size.width * dim_hog, out_size.height), CV_64F);

    const size_t im_stride = image.step1();
    const size_t hist_stride = hist.step1();
    const size_t norm_stride = norm.step1();
    const size_t feat_stride = features.step1();

    const double *im = image.ptr<double>(0);
    double *hist_ptr = hist.ptr<double>(0);
    double *norm_ptr = norm.ptr<double>(0);
    double *feat_ptr = features.ptr<double>(0);

    for (int y = 1; y < visible.height - 1; ++y) {
        for (int x = 1; x < visible.width - 1; ++x) {
            const double *s = im + 3 * std::min(x, image.cols - 2) + std::min(y, image.rows - 2) * im_stride;

            double dyb = *(s + im_stride) - *(s - im_stride);
            double dxb = *(s + 3) - *(s - 3);
            double vb = dxb * dxb + dyb * dyb;

            s += 1;
            double dyg = *(s + im_stride) - *(s - im_stride);
            double dxg = *(s + 3) - *(s - 3);
            double vg = dxg * dxg + dyg * dyg;

            s += 1;
            double dy = *(s + im_stride) - *(s - im_stride);
            double dx = *(s + 3) - *(s - 3);
            double v = dx * dx + dy * dy;

            if (vg > v) {
                v = vg;
                dx = dxg;
                dy = dyg;
            }
            if (vb > v) {
                v = vb;
                dx = dxb;
                dy = dyb;
            }

            double best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < num_orient / 2; ++o) {
                double dot = uu[o] * dx + vv[o] * dy;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                } else if (-dot > best_dot) {
                    best_dot = -dot;
                    best_o = o + num_orient / 2;
                }
            }

            double yp = (static_cast<double>(y) + 0.5) / static_cast<double>(bin_size) - 0.5;
            double xp = (static_cast<double>(x) + 0.5) / static_cast<double>(bin_size) - 0.5;
            int iyp = static_cast<int>(cvFloor(yp));
            int ixp = static_cast<int>(cvFloor(xp));
            double vy0 = yp - iyp;
            double vx0 = xp - ixp;
            double vy1 = 1.0 - vy0;
            double vx1 = 1.0 - vx0;
            v = std::sqrt(v);

            if (iyp >= 0 && ixp >= 0) {
                *(hist_ptr + iyp * hist_stride + ixp * num_orient + best_o) += vy1 * vx1 * v;
            }
            if (iyp >= 0 && ixp + 1 < block_size.width) {
                *(hist_ptr + iyp * hist_stride + (ixp + 1) * num_orient + best_o) += vx0 * vy1 * v;
            }
            if (iyp + 1 < block_size.height && ixp >= 0) {
                *(hist_ptr + (iyp + 1) * hist_stride + ixp * num_orient + best_o) += vy0 * vx1 * v;
            }
            if (iyp + 1 < block_size.height && ixp + 1 < block_size.width) {
                *(hist_ptr + (iyp + 1) * hist_stride + (ixp + 1) * num_orient + best_o) += vy0 * vx0 * v;
            }
        }
    }

    for (int y = 0; y < block_size.height; ++y) {
        const double *src = hist_ptr + y * hist_stride;
        double *dst = norm_ptr + y * norm_stride;
        double *dst_end = dst + block_size.width;
        while (dst < dst_end) {
            *dst = 0.0;
            for (int o = 0; o < num_orient / 2; ++o) {
                *dst += (*src + *(src + num_orient / 2)) * (*src + *(src + num_orient / 2));
                ++src;
            }
            ++dst;
            src += num_orient / 2;
        }
    }

    for (int y = pad_y; y < out_size.height - pad_y; ++y) {
        for (int x = pad_x; x < out_size.width - pad_x; ++x) {
            double *dst = feat_ptr + y * feat_stride + x * dim_hog;
            double *p = nullptr;
            double n1 = 0.0;
            double n2 = 0.0;
            double n3 = 0.0;
            double n4 = 0.0;
            const double *src = nullptr;

            p = norm_ptr + (y - pad_y + 1) * norm_stride + (x - pad_x + 1);
            n1 = 1.0 / std::sqrt(*p + *(p + 1) + *(p + norm_stride) + *(p + norm_stride + 1) + eps);
            p = norm_ptr + (y - pad_y) * norm_stride + (x - pad_x + 1);
            n2 = 1.0 / std::sqrt(*p + *(p + 1) + *(p + norm_stride) + *(p + norm_stride + 1) + eps);
            p = norm_ptr + (y - pad_y + 1) * norm_stride + x - pad_x;
            n3 = 1.0 / std::sqrt(*p + *(p + 1) + *(p + norm_stride) + *(p + norm_stride + 1) + eps);
            p = norm_ptr + (y - pad_y) * norm_stride + x - pad_x;
            n4 = 1.0 / std::sqrt(*p + *(p + 1) + *(p + norm_stride) + *(p + norm_stride + 1) + eps);

            double t1 = 0.0;
            double t2 = 0.0;
            double t3 = 0.0;
            double t4 = 0.0;

            src = hist_ptr + (y - pad_y + 1) * hist_stride + (x - pad_x + 1) * num_orient;
            for (int o = 0; o < num_orient; ++o) {
                double val = *src;
                double h1 = std::min(val * n1, 0.2);
                double h2 = std::min(val * n2, 0.2);
                double h3 = std::min(val * n3, 0.2);
                double h4 = std::min(val * n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);
                ++src;
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
            }

            src = hist_ptr + (y - pad_y + 1) * hist_stride + (x - pad_x + 1) * num_orient;
            for (int o = 0; o < num_orient / 2; ++o) {
                double sum = *src + *(src + num_orient / 2);
                double h1 = std::min(sum * n1, 0.2);
                double h2 = std::min(sum * n2, 0.2);
                double h3 = std::min(sum * n3, 0.2);
                double h4 = std::min(sum * n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);
                ++src;
            }

            *(dst++) = 0.2357 * t1;
            *(dst++) = 0.2357 * t2;
            *(dst++) = 0.2357 * t3;
            *(dst++) = 0.2357 * t4;
            *dst = 0.0;
        }
    }

    for (int y = 0; y < features.rows; ++y) {
        for (int x = 0; x < features.cols; x += dim_hog) {
            if (y > pad_y - 1 && y < features.rows - pad_y &&
                x > pad_x * dim_hog - 1 && x < features.cols - pad_x * dim_hog) {
                continue;
            }
            features.at<double>(y, x + dim_hog - 1) = 1.0;
        }
    }
}

std::vector<cv::Mat> GetFeaturesHog(const cv::Mat &image, int bin_size) {
    cv::Mat hog_matrix;
    cv::Mat img_double;
    image.convertTo(img_double, CV_64FC3, 1.0 / 255.0);
    ComputeHog32D(img_double, hog_matrix, bin_size, 1, 1);
    hog_matrix.convertTo(hog_matrix, CV_32F);

    cv::Size hog_size = image.size();
    hog_size.width /= bin_size;
    hog_size.height /= bin_size;

    cv::Mat hog_channels(hog_size, CV_32FC(32), hog_matrix.data);
    std::vector<cv::Mat> features;
    cv::split(hog_channels, features);
    return features;
}

std::vector<cv::Mat> GetFeaturesCn(const cv::Mat &image, const cv::Size &output_size) {
    cv::Mat patch = image.clone();
    cv::Mat cn_features = cv::Mat::zeros(patch.rows, patch.cols, CV_32FC(10));

    for (int y = 0; y < patch.rows; ++y) {
        for (int x = 0; x < patch.cols; ++x) {
            const cv::Vec3b &pixel = patch.at<cv::Vec3b>(y, x);
            unsigned index = static_cast<unsigned>(
                cvFloor(static_cast<float>(pixel[2]) / 8.0f) +
                32 * cvFloor(static_cast<float>(pixel[1]) / 8.0f) +
                32 * 32 * cvFloor(static_cast<float>(pixel[0]) / 8.0f));

            for (int k = 0; k < 10; ++k) {
                cn_features.at<cv::Vec<float, 10> >(y, x)[k] = kColorNames[index][k];
            }
        }
    }

    std::vector<cv::Mat> result;
    cv::split(cn_features, result);
    for (size_t i = 0; i < result.size(); ++i) {
        if (output_size.width > 0 && output_size.height > 0) {
            cv::resize(result[i], result[i], output_size, 0, 0, cv::INTER_CUBIC);
        }
    }

    return result;
}

std::vector<cv::Mat> GetFeaturesRgb(const cv::Mat &patch, const cv::Size &output_size) {
    std::vector<cv::Mat> channels;
    cv::split(patch, channels);
    for (size_t i = 0; i < channels.size(); ++i) {
        channels[i].convertTo(channels[i], CV_32F, 1.0 / 255.0, -0.5);
        channels[i] = channels[i] - cv::mean(channels[i])[0];
        cv::resize(channels[i], channels[i], output_size, 0, 0, cv::INTER_CUBIC);
    }
    return channels;
}

cv::Mat BgrToHsv(const cv::Mat &image) {
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv_image, channels);
    channels.at(0).convertTo(channels.at(0), CV_8UC1, 255.0 / 180.0);
    cv::merge(channels, hsv_image);
    return hsv_image;
}

}  // namespace csrt
