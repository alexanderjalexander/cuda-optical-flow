#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <linux/limits.h>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "gpu/lk.cuh"

struct VideoInfo
{
    std::vector<cv::Mat> frames;
    double fps;
    int totalFrames;
    int width;
    int height;
};

int
readVideo(VideoInfo &video, std::filesystem::path video_path)
{
    cv::VideoCapture cap(video_path.string());

    video.totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    video.fps = cap.get(cv::CAP_PROP_FPS);

    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video file." << std::endl;
        cap.release();
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Video file opened successfully!" << std::endl;
        std::cout << "Total frames in video: " << video.totalFrames << std::endl;
        std::cout << "Video FPS: " << video.fps << std::endl;
    }

    cv::Mat frame, gray;
    video.frames.reserve(video.totalFrames > 0 ? video.totalFrames : 100);

    while (cap.read(frame))
    {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        video.frames.push_back(gray.clone());
    }
    cap.release();
    std::cout << "Buffered " << video.frames.size() << " frames." << std::endl;

    return EXIT_SUCCESS;
}

int
writeVideo(VideoInfo &video, std::filesystem::path video_path)
{
    if (video.frames.empty())
    {
        std::cerr << "Error: No frames to write." << std::endl;
        return EXIT_FAILURE;
    }

    int width  = video.frames[0].cols;
    int height = video.frames[0].rows;
    bool isColor = (video.frames[0].channels() == 3);

    cv::VideoWriter writer(
        video_path.string(),
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
        video.fps,
        cv::Size(width, height),
        isColor
    );

    if (!writer.isOpened())
    {
        std::cerr << "Error: Could not open output video for writing: " << video_path << std::endl;
        writer.release();
        return EXIT_FAILURE;
    }

    cv::Mat writeFrame;
    for (size_t i = 0; i < video.frames.size(); ++i)
    {
        if (video.frames[i].depth() != CV_8U)
        {
            cv::normalize(video.frames[i], writeFrame, 0, 255, cv::NORM_MINMAX, CV_8U);
        }
        else
        {
            writeFrame = video.frames[i];
        }

        writer.write(writeFrame);

        if ((i + 1) % 50 == 0 || i == 0)
        {
            std::cout << "Frames Written: " << (i + 1) << " / " << video.frames.size() << std::endl;
        }
    }

    writer.release();
    std::cout << "Wrote " << video.frames.size() << " frames to: " << video_path << std::endl;
    return EXIT_SUCCESS;
}

int
main(int argc, char *argv[])
{
    /**
     * 1. Read the video and decode it
     * 2. Put video frames into GPU memory
     * 3. Run kernels individually as follows
     *   - Convert to grayscale
     *   - Corner Detector, obtain features
     *   - Spatial Derivative
     *   -
     *
     *
     * https://opencv.org/reading-and-writing-videos-using-opencv/
     */

    std::filesystem::path current_dir = std::filesystem::current_path();
    std::filesystem::path example_file = "Example-Video.mp4";
    std::filesystem::path full_path = current_dir / "Example-Video.mp4";
    std::filesystem::path output_path = current_dir / "Example-Video-Output.mp4";

    
    VideoInfo video;
    if (readVideo(video, full_path) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    std::cout << std::endl << "Sending frames to GPU..." << std::endl;
    std::cout << "Frames to Process: " << video.frames.size() << std::endl;
    if (!video.frames.empty())
    {
        int width = video.frames[0].cols;
        int height = video.frames[0].rows;
        size_t framePixels = width * height;

        cv::Mat result;

        for (size_t i = 0; i < video.frames.size(); ++i)
        {
            // TODO: Flesh this out
            lucasKanadeSingleFrameGPU(video.frames[i], result);
            
            if ((i + 1) % 50 == 0 || i == 0)
            {
                std::cout << "Frames Processed: " << (i + 1) << " / " << video.frames.size() << std::endl;
            }

            video.frames[i] = result.clone();
        }

        std::cout << "GPU processing complete! Processed all " << video.frames.size() << " frames." << std::endl;
    }

    std::cout << std::endl << "Writing output video..." << std::endl;

    if (writeVideo(video, output_path) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}