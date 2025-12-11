#ifndef UPDATE_CSRT_DCF_SOLVER_HPP
#define UPDATE_CSRT_DCF_SOLVER_HPP

#include <opencv2/opencv.hpp>
#include <complex>
#include "Config.hpp"

namespace update_csrt {

/**
 * @brief DCF Solver with ADMM and Mask Constraint
 * 
 * Solves the optimization problem:
 * min_h ||y - F⊙h||² + λ||h||²
 * s.t. h = m⊙h  (mask constraint)
 * 
 * Uses ADMM (Alternating Direction Method of Multipliers)
 */
class DCFSolver {
public:
    /**
     * @brief Constructor
     * @param config Configuration object
     */
    explicit DCFSolver(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~DCFSolver();
    
    /**
     * @brief Solve for correlation filter (unconstrained)
     * @param features Training features F (C x H x W)
     * @param label Desired response y (H x W)
     * @return Filter h (C x H x W)
     */
    cv::Mat solveUnconstrained(const cv::Mat& features, const cv::Mat& label);
    
    /**
     * @brief Solve with binary mask constraint using ADMM
     * @param features Training features F (C x H x W)
     * @param label Desired response y (H x W)
     * @param mask Binary mask m (H x W, 0 or 1)
     * @return Constrained filter h (C x H x W)
     */
    cv::Mat solveWithMask(const cv::Mat& features, 
                          const cv::Mat& label, 
                          const cv::Mat& mask);
    
    /**
     * @brief Apply filter to get response map
     * @param filter Filter h (C x H x W)
     * @param features Input features F (C x H x W)
     * @return Response map (H x W)
     */
    cv::Mat applyFilter(const cv::Mat& filter, const cv::Mat& features);
    
    /**
     * @brief Create Gaussian label (desired response)
     * @param size Output size (H, W)
     * @param sigma Gaussian sigma
     * @return Gaussian peak (H x W)
     */
    cv::Mat createGaussianLabel(const cv::Size& size, float sigma);

private:
    Config config_;
    
    /**
     * @brief FFT helper - forward transform
     */
    cv::Mat fft2(const cv::Mat& input);
    
    /**
     * @brief FFT helper - inverse transform
     */
    cv::Mat ifft2(const cv::Mat& input);
    
    /**
     * @brief Complex multiplication
     */
    cv::Mat complexMultiply(const cv::Mat& a, const cv::Mat& b);
    
    /**
     * @brief Complex division
     */
    cv::Mat complexDivide(const cv::Mat& a, const cv::Mat& b);
    
    /**
     * @brief Complex conjugate
     */
    cv::Mat complexConjugate(const cv::Mat& a);
    
    /**
     * @brief ADMM iteration step
     */
    cv::Mat admm_step(const cv::Mat& features_fft,
                      const cv::Mat& label_fft,
                      const cv::Mat& mask,
                      const cv::Mat& h_prev,
                      const cv::Mat& z,
                      const cv::Mat& u);
};

} // namespace update_csrt

#endif // UPDATE_CSRT_DCF_SOLVER_HPP
