#ifndef VIDEO_IO_H
#define VIDEO_IO_H

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <vector>
#include <memory>

#include "frame_buffer.hpp"
#define FRAME_BUFFER_SIZE 10

struct VideoInfo
{
    FrameBuffer frames;
    std::vector<cv::Mat> outputFrames;
    double fps;
    int width;
    int height;
};

int readVideo(VideoInfo &video, std::filesystem::path video_path, bool usePinnedMemory);
int writeVideo(VideoInfo &video, std::filesystem::path video_path);
int copyVideo(VideoInfo &video1, VideoInfo &video2);

#endif
