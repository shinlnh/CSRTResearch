#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace csrt {

int Mod(int a, int b);

double KernelEpan(double x);

cv::Mat CircShift(const cv::Mat &matrix, int dx, int dy);

cv::Mat GaussianShapedLabels(float sigma, int width, int height);

std::vector<cv::Mat> FourierTransformFeatures(const std::vector<cv::Mat> &features);

cv::Mat DivideComplexMatrices(const cv::Mat &a, const cv::Mat &b);

cv::Mat GetSubwindow(const cv::Mat &image, const cv::Point2f &center, int width, int height,
    cv::Rect *valid_pixels = nullptr);

float SubpixelPeak(const cv::Mat &response, const std::string &direction, const cv::Point2f &peak);

double GetMax(const cv::Mat &matrix);

double GetMin(const cv::Mat &matrix);

cv::Mat GetHannWindow(cv::Size size);

cv::Mat GetKaiserWindow(cv::Size size, float alpha);

cv::Mat GetChebyshevWindow(cv::Size size, float attenuation);

std::vector<cv::Mat> GetFeaturesRgb(const cv::Mat &patch, const cv::Size &output_size);

std::vector<cv::Mat> GetFeaturesHog(const cv::Mat &image, int bin_size);

std::vector<cv::Mat> GetFeaturesCn(const cv::Mat &image, const cv::Size &output_size);

cv::Mat BgrToHsv(const cv::Mat &image);

}  // namespace csrt
