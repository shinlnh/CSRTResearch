#include "../inc/HOGExtractor.hpp"
#include <iostream>
#include <cmath>

namespace update_csrt {

HOGExtractor::HOGExtractor(const Config& config)
    : config_(config), hog_channels_(21), cn_channels_(10) {
    
    num_channels_ = hog_channels_ + cn_channels_;  // 31 total
    
    // Initialize ColorNames lookup table
    initColorNamesTable();
    
    std::cout << "HOGExtractor initialized: " 
              << hog_channels_ << " HOG + " 
              << cn_channels_ << " ColorNames = " 
              << num_channels_ << " channels" << std::endl;
}

HOGExtractor::~HOGExtractor() {
    // Nothing to cleanup
}

void HOGExtractor::initColorNamesTable() {
    // Simplified ColorNames initialization
    // In full implementation, this would load from pre-computed table
    // For now, create dummy table
    cn_table_ = cv::Mat::zeros(256 * 256 * 256, cn_channels_, CV_32F);
    
    // Simple heuristic: map colors to 10 color names
    // (black, blue, brown, grey, green, orange, pink, purple, red, white, yellow)
    // This is a simplified version - real ColorNames use learned table
}

cv::Mat HOGExtractor::computeHOG(const cv::Mat& image) {
    // Convert to grayscale
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Compute gradients
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    
    // Compute magnitude and orientation
    cv::Mat magnitude, orientation;
    cv::cartToPolar(grad_x, grad_y, magnitude, orientation, true);  // Angles in degrees
    
    // Quantize orientation into bins
    int n_bins = config_.hog_orientations;  // 9 bins
    float bin_size = 180.0f / n_bins;  // 20 degrees per bin
    
    // For simplicity, use OpenCV's HOGDescriptor
    cv::HOGDescriptor hog(
        cv::Size(image.cols, image.rows),  // win_size
        cv::Size(config_.hog_block_size * config_.hog_cell_size, 
                 config_.hog_block_size * config_.hog_cell_size),  // block_size
        cv::Size(config_.hog_cell_size, config_.hog_cell_size),  // block_stride
        cv::Size(config_.hog_cell_size, config_.hog_cell_size),  // cell_size
        n_bins  // nbins
    );
    
    std::vector<float> descriptors;
    hog.compute(gray, descriptors);
    
    // Reshape to spatial feature map (simplified)
    int H = image.rows / config_.hog_cell_size;
    int W = image.cols / config_.hog_cell_size;
    
    // Create multi-channel HOG feature map (21 channels)
    cv::Mat hog_features = cv::Mat::zeros(hog_channels_, H, W, CV_32F);
    
    // Fill with computed HOG descriptors (simplified mapping)
    int descriptor_idx = 0;
    for (int c = 0; c < hog_channels_ && descriptor_idx < descriptors.size(); ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (descriptor_idx < descriptors.size()) {
                    hog_features.at<float>(c, y, x) = descriptors[descriptor_idx++];
                }
            }
        }
    }
    
    return hog_features;
}

cv::Mat HOGExtractor::computeColorNames(const cv::Mat& image) {
    // Resize to feature map size
    int H = image.rows / config_.hog_cell_size;
    int W = image.cols / config_.hog_cell_size;
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(W, H));
    
    // Create ColorNames feature map (10 channels)
    std::vector<cv::Mat> cn_channels(cn_channels_);
    for (int i = 0; i < cn_channels_; ++i) {
        cn_channels[i] = cv::Mat::zeros(H, W, CV_32F);
    }
    
    // Simplified ColorNames mapping (heuristic-based)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            
            // Simple heuristic to map to color names
            float intensity = (r + g + b) / 3.0f;
            float saturation = std::max({r, g, b}) - std::min({r, g, b});
            
            // Distribute probability across color name channels
            // Channel 0: Black
            cn_channels[0].at<float>(y, x) = (intensity < 50) ? 1.0f : 0.0f;
            
            // Channel 1: White  
            cn_channels[1].at<float>(y, x) = (intensity > 200 && saturation < 30) ? 1.0f : 0.0f;
            
            // Channel 2: Red
            cn_channels[2].at<float>(y, x) = (r > g && r > b && saturation > 50) ? 1.0f : 0.0f;
            
            // Channel 3: Green
            cn_channels[3].at<float>(y, x) = (g > r && g > b && saturation > 50) ? 1.0f : 0.0f;
            
            // Channel 4: Blue
            cn_channels[4].at<float>(y, x) = (b > r && b > g && saturation > 50) ? 1.0f : 0.0f;
            
            // Channels 5-9: Other colors (simplified)
            // Grey, Brown, Orange, Pink, Purple
            for (int i = 5; i < cn_channels_; ++i) {
                cn_channels[i].at<float>(y, x) = 0.1f;  // Uniform prior
            }
        }
    }
    
    // Merge channels into single tensor
    cv::Mat cn_features;
    cv::merge(cn_channels, cn_features);
    
    // Reshape to (C, H, W)
    std::vector<int> sizes = {cn_channels_, H, W};
    cn_features = cn_features.reshape(1, sizes);
    
    return cn_features;
}

cv::Mat HOGExtractor::extract(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Empty input image" << std::endl;
        return cv::Mat();
    }
    
    // Compute HOG features (21 channels)
    cv::Mat hog_feat = computeHOG(image);
    
    // Compute ColorNames features (10 channels)
    cv::Mat cn_feat = computeColorNames(image);
    
    if (hog_feat.empty() || cn_feat.empty()) {
        std::cerr << "Feature extraction failed" << std::endl;
        return cv::Mat();
    }
    
    // Concatenate along channel dimension
    // HOG: (21, H, W), CN: (10, H, W) → (31, H, W)
    int H = hog_feat.size[1];
    int W = hog_feat.size[2];
    
    std::vector<int> sizes = {num_channels_, H, W};
    cv::Mat features(sizes, CV_32F);
    
    // Copy HOG channels (0-20)
    for (int c = 0; c < hog_channels_; ++c) {
        cv::Range src_ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Range dst_ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        hog_feat(src_ranges).copyTo(features(dst_ranges));
    }
    
    // Copy ColorNames channels (21-30)
    for (int c = 0; c < cn_channels_; ++c) {
        cv::Range src_ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Range dst_ranges[] = {cv::Range(hog_channels_ + c, hog_channels_ + c + 1), 
                                   cv::Range::all(), cv::Range::all()};
        cn_feat(src_ranges).copyTo(features(dst_ranges));
    }
    
    return features;
}

cv::Mat HOGExtractor::applyMask(const cv::Mat& features, const cv::Mat& mask) {
    if (mask.empty() || features.empty()) {
        return features;
    }
    
    // Resize mask to match feature spatial dimensions
    int H = features.size[1];
    int W = features.size[2];
    
    cv::Mat mask_resized;
    cv::resize(mask, mask_resized, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
    
    // Convert mask to float [0, 1]
    cv::Mat mask_float;
    mask_resized.convertTo(mask_float, CV_32F, 1.0 / 255.0);
    
    // Apply mask to each channel
    cv::Mat masked_features = features.clone();
    int C = features.size[0];
    
    for (int c = 0; c < C; ++c) {
        cv::Range ranges[] = {cv::Range(c, c+1), cv::Range::all(), cv::Range::all()};
        cv::Mat channel = masked_features(ranges).clone();
        channel = channel.reshape(1, H);  // Reshape to 2D (H, W)
        
        // Multiply by mask
        channel = channel.mul(mask_float);
        
        // Put back
        channel.reshape(1, 1).copyTo(masked_features(ranges));
    }
    
    return masked_features;
}

cv::Mat HOGExtractor::extractMasked(const cv::Mat& image, const cv::Mat& mask) {
    cv::Mat features = extract(image);
    
    if (features.empty()) {
        return cv::Mat();
    }
    
    return applyMask(features, mask);
}

} // namespace update_csrt
