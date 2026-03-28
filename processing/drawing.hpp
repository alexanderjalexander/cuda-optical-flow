#ifndef DRAWING_H
#define DRAWING_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <vector>

using namespace cv;
using namespace std;

vector<Scalar> getRandomColors(int num);

void drawOpticalFlow(
    Mat &output,
    Mat &mask,
    const vector<Point2f> &p0,
    const vector<Point2f> &p1,
    const vector<uchar> &status,
    const vector<Scalar>& colors,
    bool drawContinuous
);

void drawOpticalFlowGPU(
    Mat &output,
    Mat &mask,
    Vec3f *features,
    int featureCount,
    const vector<Scalar>& colors,
    bool drawContinuous
);

#endif