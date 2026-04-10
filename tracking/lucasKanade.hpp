#ifndef LUCAS_KANADE_H
#define LUCAS_KANADE_H

#include <opencv2/opencv.hpp>

#include "../processing/video_io.hpp"

#define DRAW_CONTINUOUS_LINES true

#define MAX_FEATURES 500
#define QUALITY_LEVEL 0.01

#define SOBEL_MASK_SIZE 3

#define HARRIS_MASK_SIZE 5
#define HARRIS_EPSILON 0.04
#define HARRIS_DISTANCE 5

#define LK_EPSILON 0.03
#define LK_ITERATIONS 10

void sparseLucasKanadeCPU(VideoInfo &video);
void sparseLucasKanadeGPU(VideoInfo &video);

void denseLucasKanadeCPU(VideoInfo &video);

#endif
