#ifndef LUCAS_KANADE_H
#define LUCAS_KANADE_H

#include <opencv2/opencv.hpp>

#include "../flags.hpp"
#include "../processing/video_io.hpp"

// Good, universal block size.
#define BLOCK_SIZE 16

#define DRAW_CONTINUOUS_LINES true

#define MAX_FEATURES 500
#define QUALITY_LEVEL 0.01
#define MAX_PYR_LEVELS 3

#define SOBEL_MASK_SIZE 3
#define SOBEL_MASK_RAD (SOBEL_MASK_SIZE / 2)

#define HARRIS_MASK_SIZE 5
#define HARRIS_MASK_RAD (HARRIS_MASK_SIZE / 2)
#define HARRIS_EPSILON 0.04
#define HARRIS_DISTANCE 5

#define LK_WINDOW_WIDTH 15
#define LK_WINDOW_WIDTH_HALF (LK_WINDOW_WIDTH / 2)
#define LK_WINDOW_NUM (LK_WINDOW_WIDTH * LK_WINDOW_WIDTH)

#define LK_EPSILON 0.03
#define LK_ITERATIONS 10

static uint64 correctFlows = 0;
static uint64 totalFlows = 0;

void compareCPUSingleFlowOnDense(cv::Mat oldFrame, cv::Mat frame, vector<cv::Point2f> oldPoints,
                                 vector<cv::Point2f> points, vector<uchar> status, ProgramFlags flags);
void compareGPUSingleFlowOnDense(cv::Mat oldFrame, cv::Mat frame, cv::Vec3f *oldPoints, cv::Vec3f *points,
                                 int featureCount, ProgramFlags flags);

void sparseLucasKanadeCPU(VideoInfo &video, ProgramFlags flags);

void sparseLucasKanadeGPU(VideoInfo &video, ProgramFlags flags);
void sparseLucasKanadeGPUTex(VideoInfo &video, ProgramFlags flags);
void sparseLucasKanadeGPUMip(VideoInfo &video, ProgramFlags flags);

void denseLucasKanadeCPU(VideoInfo &video, ProgramFlags flags);

#endif
