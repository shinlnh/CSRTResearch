#ifndef UPDATE_CSRT_CONFIG_HPP
#define UPDATE_CSRT_CONFIG_HPP

#include <string>
#include <opencv2/opencv.hpp>

namespace update_csrt {

/**
 * @brief Configuration for Updated CSRT Tracker with Deep Features
 * 
 * This configuration includes all hyperparameters for:
 * - Deep feature extraction (VGG16)
 * - CorrProject network projection
 * - Adaptive gating mechanism
 * - ADMM solver parameters
 * - Loss weights and constraints
 */
struct Config {
    // ==================== Deep Feature Parameters ====================
    
    // Backbone network settings
    std::string backbone = "vgg16";           // Feature extraction backbone
    std::string feature_layer = "conv4_3";    // Which layer to extract
    int deep_feature_channels = 512;          // VGG16 conv4_3 channels
    
    // ONNX model paths
    std::string vgg16_onnx_path = "update_csrt/models/vgg16_conv4_3.onnx";
    std::string corr_project_onnx_path = "update_csrt/models/corr_project.onnx";
    std::string adaptive_gate_onnx_path = "update_csrt/models/adaptive_gating.onnx";
    
    // ==================== Tracker Window Size ====================
    
    cv::Size template_size = cv::Size(127, 127);  // Template crop size
    cv::Size search_size = cv::Size(255, 255);    // Search region size
    float padding = 2.0f;                         // Context padding ratio
    
    // ==================== DCF Filter Parameters ====================
    
    int num_channels = 31;                    // Output correlation filter channels
    float learning_rate = 0.025f;             // Filter update learning rate
    float regularization = 1e-4f;             // Tikhonov regularization λ
    
    // ==================== ADMM Solver Parameters ====================
    
    int admm_iterations = 4;                  // ADMM iterations per frame
    float admm_rho = 1.0f;                    // ADMM penalty parameter ρ
    float admm_mu = 100.0f;                   // Augmented Lagrangian μ
    bool use_mask_constraint = true;          // Enable h = m⊙h constraint
    
    // ==================== Adaptive Gating ====================
    
    bool use_adaptive_alpha = true;           // Enable learned alpha
    float alpha_min = 0.3f;                   // Min blending weight
    float alpha_max = 0.9f;                   // Max blending weight
    float alpha_default = 0.6f;               // Default if not adaptive
    
    // ==================== Binary Mask Generation ====================
    
    float mask_threshold = 0.5f;              // Response threshold for mask
    int mask_morph_size = 5;                  // Morphological kernel size
    bool use_deep_mask = true;                // Use deep features for mask
    
    // ==================== Spatial Reliability ====================
    
    int spatial_bins = 16;                    // Spatial grid bins (16x16)
    float spatial_sigma = 0.5f;               // Gaussian falloff σ
    bool learn_spatial_weights = true;        // Learn from deep features
    
    // ==================== Channel Reliability ====================
    
    bool use_channel_weights = true;          // Enable channel weighting
    bool learn_channel_weights = true;        // Learn from deep features
    float channel_reg = 0.01f;                // Channel weight regularization
    
    // ==================== Rescue Strategy ====================
    
    bool use_rescue = true;                   // Enable failure recovery
    float rescue_threshold = 0.15f;           // PSR threshold for rescue
    int rescue_history_size = 10;             // History buffer size
    float rescue_similarity_threshold = 0.7f; // Deep feature similarity
    
    // ==================== Loss Weights (for training reference) ====================
    
    float loss_peak = 1.0f;                   // Peak loss weight λ₁
    float loss_smooth = 0.1f;                 // Smoothness loss λ₂
    float loss_reg = 1e-4f;                   // Regularization loss λ₃
    
    // ==================== Visualization & Debugging ====================
    
    bool visualize = true;                    // Show tracking visualization
    bool show_response_map = false;           // Display response heatmap
    bool show_mask = false;                   // Display binary mask
    bool verbose = true;                      // Print debug info
    
    // ==================== Helper Methods ====================
    
    /**
     * @brief Print all configuration parameters
     */
    void print() const {
        std::cout << "================================================================================\n";
        std::cout << "Updated CSRT Tracker Configuration\n";
        std::cout << "================================================================================\n";
        std::cout << "Backbone: " << backbone << " (" << feature_layer << ")\n";
        std::cout << "Deep channels: " << deep_feature_channels << " -> " << num_channels << "\n";
        std::cout << "Template size: " << template_size << "\n";
        std::cout << "Search size: " << search_size << "\n";
        std::cout << "Adaptive alpha: " << (use_adaptive_alpha ? "Yes" : "No") 
                  << " [" << alpha_min << ", " << alpha_max << "]\n";
        std::cout << "ADMM iterations: " << admm_iterations << " (ρ=" << admm_rho 
                  << ", μ=" << admm_mu << ")\n";
        std::cout << "Mask constraint: " << (use_mask_constraint ? "Enabled" : "Disabled") << "\n";
        std::cout << "Rescue strategy: " << (use_rescue ? "Enabled" : "Disabled") << "\n";
        std::cout << "================================================================================\n";
    }
    
    /**
     * @brief Validate configuration parameters
     */
    bool validate() const {
        if (num_channels <= 0) {
            std::cerr << "Error: num_channels must be positive\n";
            return false;
        }
        if (alpha_min >= alpha_max) {
            std::cerr << "Error: alpha_min must be < alpha_max\n";
            return false;
        }
        if (admm_iterations <= 0) {
            std::cerr << "Error: admm_iterations must be positive\n";
            return false;
        }
        if (learning_rate <= 0 || learning_rate >= 1) {
            std::cerr << "Error: learning_rate must be in (0, 1)\n";
            return false;
        }
        return true;
    }
};

} // namespace update_csrt

#endif // UPDATE_CSRT_CONFIG_HPP
