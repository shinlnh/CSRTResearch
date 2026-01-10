#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "Build information:\n" << cv::getBuildInformation() << std::endl;
    
#ifdef HAVE_OPENCV_CUDAARITHM
    std::cout << "\n=== CUDA Support: YES ===" << std::endl;
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices found: " << cuda_devices << std::endl;
    if (cuda_devices > 0) {
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    }
#else
    std::cout << "\n=== CUDA Support: NO ===" << std::endl;
#endif
    
    return 0;
}
