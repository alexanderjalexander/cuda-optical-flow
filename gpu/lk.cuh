#ifndef LK_H
#define LK_H

#include <opencv2/opencv.hpp>

void lucasKanade(const cv::Mat &prevFrame, const cv::Mat &frame, cv::Mat &result, int maxFeatures);

#endif
