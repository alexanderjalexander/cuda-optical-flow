#ifndef LUCAS_KANADE_H
#define LUCAS_KANADE_H

#include <opencv2/opencv.hpp>

#include "../processing/video_io.hpp"

#define DRAW_CONTINUOUS_LINES true
#define MAX_FEATURES 500

#define HARRIS_EPSILON 0.04
#define HARRIS_DISTANCE 10
#define MAX_LK_ITERATIONS 10
#define LK_EPSILON 0.03f

void sparseLucasKanadeCPU(VideoInfo &video);
void sparseLucasKanadeGPU(VideoInfo &video);

#endif
