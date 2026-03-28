#ifndef VIDEO_IO_H
#define VIDEO_IO_H

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <vector>

struct VideoInfo
{
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> outputFrames;
    double fps;
    int width;
    int height;
};

int readVideo(VideoInfo &video, std::filesystem::path video_path);
int writeVideo(VideoInfo &video, std::filesystem::path video_path);
int copyVideo(VideoInfo &video1, VideoInfo &video2);

#endif
