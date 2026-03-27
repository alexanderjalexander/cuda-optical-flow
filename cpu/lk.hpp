#ifndef LK_CPU_H
#define LK_CPU_H

#include <opencv2/opencv.hpp>

void lucasKanadeCPU(const cv::Mat &prevFrame, const cv::Mat &frame, cv::Mat &result, int maxFeatures);

#endif
