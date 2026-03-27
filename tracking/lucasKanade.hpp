#ifndef LUCAS_KANADE_H
#define LUCAS_KANADE_H

#include <opencv2/opencv.hpp>

#include "../processing/video_io.hpp"

#define MAX_FEATURES 40000

void lucasKanadeCPU(VideoInfo &video);
void lucasKanadeGPU(VideoInfo &video);

#endif