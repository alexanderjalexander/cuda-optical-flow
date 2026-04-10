#ifndef DRAWING_H
#define DRAWING_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <vector>

std::vector<cv::Scalar> getRandomColors(int num);

void drawSparseOpticalFlow(
    cv::Mat &output,
    cv::Mat &mask,
    const std::vector<cv::Point2f> &p0,
    const std::vector<cv::Point2f> &p1,
    const std::vector<uchar> &status,
    const std::vector<cv::Scalar>& colors,
    bool drawContinuous
);

void drawDenseOpticalFlow(
    cv::Mat &output,
    const std::vector<cv::Point2f> &p0,
    const std::vector<cv::Point2f> &p1,
    const std::vector<uchar> &status
);

void drawSparseOpticalFlowGPU(
    cv::Mat &output,
    cv::Mat &mask,
    cv::Vec3f *prevFeatures,
    cv::Vec3f *features,
    int featureCount,
    const std::vector<cv::Scalar>& colors,
    bool drawContinuous
);

#endif
