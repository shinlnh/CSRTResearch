#include "../inc/DCFSolver.hpp"
#include <iostream>
#include <cmath>

namespace update_csrt {

DCFSolver::DCFSolver(const Config& config)
    : config_(config) {
    std::cout << "DCFSolver initialized with ADMM iterations=" 
              << config_.admm_iterations << ", λ=" << config_.regularization << std::endl;
}

DCFSolver::~DCFSolver() {
    // Nothing to cleanup
}

cv::Mat DCFSolver::fft2(const cv::Mat& input) {
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(input.rows);
    int n = cv::getOptimalDFTSize(input.cols);
    cv::copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, 
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    cv::Mat complex_input;
    if (padded.channels() == 1) {
        cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
        cv::merge(planes, 2, complex_input);
    } else {
        complex_input = padded;
    }
    
    cv::Mat result;
    cv::dft(complex_input, result, cv::DFT_COMPLEX_OUTPUT);
    
    return result;
}

cv::Mat DCFSolver::ifft2(const cv::Mat& input) {
    cv::Mat result;
    cv::idft(input, result, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
    return result;
}

cv::Mat DCFSolver::complexMultiply(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat result(a.size(), CV_32FC2);
    
    for (int y = 0; y < a.rows; ++y) {
        for (int x = 0; x < a.cols; ++x) {
            cv::Vec2f a_val = a.at<cv::Vec2f>(y, x);
            cv::Vec2f b_val = b.at<cv::Vec2f>(y, x);
            
            float real = a_val[0] * b_val[0] - a_val[1] * b_val[1];
            float imag = a_val[0] * b_val[1] + a_val[1] * b_val[0];
            
            result.at<cv::Vec2f>(y, x) = cv::Vec2f(real, imag);
        }
    }
    
    return result;
}

cv::Mat DCFSolver::complexDivide(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat result(a.size(), CV_32FC2);
    
    for (int y = 0; y < a.rows; ++y) {
        for (int x = 0; x < a.cols; ++x) {
            cv::Vec2f a_val = a.at<cv::Vec2f>(y, x);
            cv::Vec2f b_val = b.at<cv::Vec2f>(y, x);
            
            float denom = b_val[0] * b_val[0] + b_val[1] * b_val[1] + 1e-8f;
            float real = (a_val[0] * b_val[0] + a_val[1] * b_val[1]) / denom;
            float imag = (a_val[1] * b_val[0] - a_val[0] * b_val[1]) / denom;
            
            result.at<cv::Vec2f>(y, x) = cv::Vec2f(real, imag);
        }
    }
    
    return result;
}

cv::Mat DCFSolver::complexConjugate(const cv::Mat& a) {
    cv::Mat result(a.size(), CV_32FC2);
    
    for (int y = 0; y < a.rows; ++y) {
        for (int x = 0; x < a.cols; ++x) {
            cv::Vec2f val = a.at<cv::Vec2f>(y, x);
            result.at<cv::Vec2f>(y, x) = cv::Vec2f(val[0], -val[1]);
        }
    }
    
    return result;
}

cv::Mat DCFSolver::createGaussianLabel(const cv::Size& size, float sigma) {
    cv::Mat label(size, CV_32F);
    
    cv::Point2f center(size.width / 2.0f, size.height / 2.0f);
    float sigma_sq = sigma * sigma;
    
    for (int y = 0; y < size.height; ++y) {
        for (int x = 0; x < size.width; ++x) {
            float dx = x - center.x;
            float dy = y - center.y;
            float dist_sq = dx * dx + dy * dy;
            
            label.at<float>(y, x) = std::exp(-dist_sq / (2.0f * sigma_sq));
        }
    }
    
    return label;
}

cv::Mat DCFSolver::solveUnconstrained(const cv::Mat& features, const cv::Mat& label) {
    if (features.empty() || label.empty()) {
        return cv::Mat();
    }
    
    int C = features.size[0];
    int H = features.size[1];
    int W = features.size[2];
    
    // Compute FFT of label
    cv::Mat label_fft = fft2(label);
    
    // Solve in frequency domain: h = F^H · y / (F^H · F + λ)
    cv::Mat filter_freq = cv::Mat::zeros(H, W, CV_32FC2);
    
    for (int c = 0; c < C; ++c) {
        // Extract channel
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = features(ranges).clone();
        channel = channel.reshape(1, H);
        
        // FFT of features
        cv::Mat channel_fft = fft2(channel);
        
        // Numerator: F^H · y
        cv::Mat conj_channel = complexConjugate(channel_fft);
        cv::Mat numerator = complexMultiply(conj_channel, label_fft);
        
        // Denominator: F^H · F + λ
        cv::Mat autocorr = complexMultiply(conj_channel, channel_fft);
        
        // Add regularization
        for (int y = 0; y < autocorr.rows; ++y) {
            for (int x = 0; x < autocorr.cols; ++x) {
                autocorr.at<cv::Vec2f>(y, x)[0] += config_.regularization;
            }
        }
        
        // Divide
        cv::Mat h_c = complexDivide(numerator, autocorr);
        
        filter_freq += h_c;
    }
    
    // Return 3D filter: (C, H, W) - per-channel filters
    std::vector<int> filter_sizes = {C, H, W};
    cv::Mat filter_3d(filter_sizes, CV_32F);
    
    for (int c = 0; c < C; ++c) {
        // Extract channel from features
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = features(ranges).clone().reshape(1, H);
        
        // FFT
        cv::Mat channel_fft = fft2(channel);
        cv::Mat conj_channel = complexConjugate(channel_fft);
        
        // Compute per-channel filter
        cv::Mat numerator = complexMultiply(conj_channel, label_fft);
        cv::Mat autocorr = complexMultiply(conj_channel, channel_fft);
        
        // Regularization
        for (int y = 0; y < autocorr.rows; ++y) {
            for (int x = 0; x < autocorr.cols; ++x) {
                autocorr.at<cv::Vec2f>(y, x)[0] += config_.regularization;
            }
        }
        
        cv::Mat h_c_freq = complexDivide(numerator, autocorr);
        cv::Mat h_c_spatial = ifft2(h_c_freq);
        
        std::vector<cv::Mat> planes;
        cv::split(h_c_spatial, planes);
        
        // Copy channel to 3D output - direct memory copy
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                filter_3d.at<float>(c, y, x) = planes[0].at<float>(y, x);
            }
        }
    }
    
    return filter_3d;
}

cv::Mat DCFSolver::solveWithMask(const cv::Mat& features, 
                                 const cv::Mat& label, 
                                 const cv::Mat& mask) {
    if (!config_.use_mask_constraint) {
        return solveUnconstrained(features, label);
    }
    
    // ADMM optimization
    // Variables: h (filter), z (auxiliary), u (Lagrange multiplier)
    
    int C = features.size[0];
    int H = features.size[1];
    int W = features.size[2];
    
    // Convert mask to float [0, 1]
    cv::Mat mask_float;
    if (mask.empty()) {
        mask_float = cv::Mat::ones(H, W, CV_32F);
    } else {
        cv::resize(mask, mask_float, cv::Size(W, H));
        mask_float.convertTo(mask_float, CV_32F, 1.0 / 255.0);
    }
    
    // Initialize variables
    cv::Mat h = solveUnconstrained(features, label);  // Initial solution [H,W]
    cv::Mat z = h.clone();
    cv::Mat u = cv::Mat::zeros(h.size(), CV_32F);
    
    std::cout << "ADMM init - h: " << h.size << ", z: " << z.size << ", u: " << u.size << std::endl;
    std::cout << "Features: " << features.size << ", Label: " << label.size() << ", Mask: " << mask_float.size() << std::endl;
    
    // ADMM iterations
    for (int iter = 0; iter < config_.admm_iterations; ++iter) {
        // Update h: argmin ||Fh - y||² + ρ/2 ||h - z + u||²
        cv::Mat h_temp = solveUnconstrained(features, label);
        
        std::cout << "Iter " << iter << " - h_temp: " << h_temp.size << std::endl;
        
        h = (h_temp + config_.admm_rho * (z - u)) / (1.0f + config_.admm_rho);
        
        // Update z: project onto mask constraint z = m⊙(h + u)
        // Apply mask element-wise (both are 2D)
        cv::multiply(h + u, mask_float, z);
        
        // Update u: u = u + h - z
        u = u + h - z;
    }
    
    return z;  // Return constrained solution
}

cv::Mat DCFSolver::applyFilter(const cv::Mat& filter, const cv::Mat& features) {
    if (filter.empty() || features.empty()) {
        return cv::Mat();
    }
    
    int C = features.size[0];
    int H = features.size[1];
    int W = features.size[2];
    
    // Correlation in frequency domain
    cv::Mat filter_fft = fft2(filter);
    cv::Mat response_freq = cv::Mat::zeros(H, W, CV_32FC2);
    
    for (int c = 0; c < C; ++c) {
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = features(ranges).clone();
        channel = channel.reshape(1, H);
        
        cv::Mat channel_fft = fft2(channel);
        
        // Correlation: F^* ⊙ h
        cv::Mat conj_channel = complexConjugate(channel_fft);
        cv::Mat corr = complexMultiply(conj_channel, filter_fft);
        
        response_freq += corr;
    }
    
    // IFFT
    cv::Mat response_spatial = ifft2(response_freq);
    
    // Extract real part
    std::vector<cv::Mat> planes;
    cv::split(response_spatial, planes);
    
    return planes[0];
}

} // namespace update_csrt
