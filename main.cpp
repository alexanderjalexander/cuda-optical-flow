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
#include "cpu/lk.hpp"
#include "timing/stopwatch.hpp"

struct VideoInfo
{
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> outputFrames;
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

    int i = 0;
    while (cap.read(frame))
    {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        video.frames.push_back(gray.clone());
        i++;
        // if ((i + 1) % 50 == 0 || i == 0)
        // {
        //     std::cout << "Frames Written: " << (i + 1) << " / " << video.frames.size() << std::endl;
        // }
    }
    cap.release();
    std::cout << "Buffered " << video.frames.size() << " frames." << std::endl;

    return EXIT_SUCCESS;
}

int
writeVideo(VideoInfo &video, std::filesystem::path video_path)
{
    if (video.outputFrames.empty())
    {
        std::cerr << "Error: No frames to write." << std::endl;
        return EXIT_FAILURE;
    }

    int width = video.outputFrames[0].cols;
    int height = video.outputFrames[0].rows;
    bool isColor = (video.outputFrames[0].channels() == 3);

    cv::VideoWriter writer(video_path.string(), cv::VideoWriter::fourcc('a', 'v', 'c', '1'), video.fps,
                           cv::Size(width, height), isColor);

    if (!writer.isOpened())
    {
        std::cerr << "Error: Could not open output video for writing: " << video_path << std::endl;
        writer.release();
        return EXIT_FAILURE;
    }

    cv::Mat writeFrame;
    for (size_t i = 0; i < video.outputFrames.size(); ++i)
    {
        if (video.outputFrames[i].depth() != CV_8U)
        {
            cv::normalize(video.outputFrames[i], writeFrame, 0, 255, cv::NORM_MINMAX, CV_8U);
        }
        else
        {
            writeFrame = video.outputFrames[i];
        }

        writer.write(writeFrame);

        // if ((i + 1) % 50 == 0 || i == 0)
        // {
        //     std::cout << "Frames Written: " << (i + 1) << " / " << video.frames.size() << std::endl;
        // }
    }

    writer.release();
    std::cout << "Wrote " << video.outputFrames.size() << " frames to: " << video_path << std::endl;
    return EXIT_SUCCESS;
}

int
main(int argc, char *argv[])
{
    std::filesystem::path current_dir = std::filesystem::current_path();
    std::string file_input = "Slow-Traffic";
    std::filesystem::path full_path = current_dir / "inputs" / (file_input + ".mp4");
    std::filesystem::path gpu_output_path = current_dir / "outputs" / (file_input + "-GPU.mp4");
    std::filesystem::path cpu_output_path = current_dir / "outputs" / (file_input + "-CPU.mp4");

    int maxFeatures = 20000;

    VideoInfo cpuVideo;
    VideoInfo gpuVideo;
    startStopwatch();
    if (readVideo(gpuVideo, full_path) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    if (readVideo(cpuVideo, full_path) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    stopStopwatch();

    // GPU Lucas Kanade

    std::cout << std::endl << "Starting GPU Lucas Kanade..." << std::endl;
    std::cout << "Frames to Process: " << gpuVideo.frames.size() << std::endl;
    startStopwatch();
    if (!gpuVideo.frames.empty())
    {
        int width = gpuVideo.frames[0].cols;
        int height = gpuVideo.frames[0].rows;
        size_t framePixels = width * height;

        cv::Mat result;

        for (size_t i = 1; i < gpuVideo.frames.size(); ++i)
        {
            // TODO: any way we can just let this run asynchronously to make it go faster?
            // Right now this just invokes LK once per frame and prev frame, and waits for it to finish.
            // TODO: either fix or remove max features
            lucasKanade(gpuVideo.frames[i - 1], gpuVideo.frames[i], result, maxFeatures);

            // if ((i + 1) % 50 == 0 || i == 0)
            // {
            //     std::cout << "Frames Processed: " << (i + 1) << " / " << gpuVideo.frames.size() << std::endl;
            // }

            gpuVideo.outputFrames.push_back(result.clone());
            if (i == 1)
            {
                // Maintain same frame count
                gpuVideo.outputFrames.push_back(result.clone());
            }
        }

        std::cout << "GPU processing complete! Processed all " << gpuVideo.frames.size() << " frames." << std::endl;
    }
    stopStopwatch();

    std::cout << std::endl << "Writing GPU Lucas Kanade output to video..." << std::endl;

    startStopwatch();
    if (writeVideo(gpuVideo, gpu_output_path) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    stopStopwatch();

    // CPU Lucas Kanade

    std::cout << std::endl << "Starting CPU Lucas Kanade..." << std::endl;
    std::cout << "Frames to Process: " << cpuVideo.frames.size() << std::endl;
    startStopwatch();
    if (!cpuVideo.frames.empty())
    {
        int width = cpuVideo.frames[0].cols;
        int height = cpuVideo.frames[0].rows;
        size_t framePixels = width * height;

        cv::Mat result;

        for (size_t i = 1; i < cpuVideo.frames.size(); ++i)
        {
            // TODO: any way we can just let this run asynchronously to make it go faster?
            // Right now this just invokes LK once per frame and prev frame, and waits for it to finish.
            // TODO: either fix or remove max features
            lucasKanadeCPU(cpuVideo.frames[i - 1], cpuVideo.frames[i], result, maxFeatures);

            // if ((i + 1) % 50 == 0 || i == 0)
            // {
            //     std::cout << "Frames Processed: " << (i + 1) << " / " << cpuVideo.frames.size() << std::endl;
            // }

            cpuVideo.outputFrames.push_back(result.clone());
            if (i == 1)
            {
                // Maintain same frame count
                cpuVideo.outputFrames.push_back(result.clone());
            }
        }

        std::cout << "CPU processing complete! Processed all " << cpuVideo.frames.size() << " frames." << std::endl;
    }
    stopStopwatch();

    std::cout << std::endl << "Writing CPU Lucas Kanade output to video..." << std::endl;

    startStopwatch();
    if (writeVideo(cpuVideo, cpu_output_path) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    stopStopwatch();

    return EXIT_SUCCESS;
}