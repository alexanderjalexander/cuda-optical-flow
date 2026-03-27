#ifndef LK_GPU_H
#define LK_GPU_H

#include <opencv2/opencv.hpp>

void lucasKanade(const cv::Mat &prevFrame, const cv::Mat &frame, cv::Mat &result, int maxFeatures);

#endif
