#ifndef LUCAS_KANADE_H
#define LUCAS_KANADE_H

#include <opencv2/opencv.hpp>

#include "../processing/video_io.hpp"

#define DRAW_CONTINUOUS_LINES true
#define MAX_FEATURES 40000

void sparseLucasKanadeCPU(VideoInfo &video);
void sparseLucasKanadeGPU(VideoInfo &video);

#endif
