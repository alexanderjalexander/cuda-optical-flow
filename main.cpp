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
#include "processing/video_io.hpp"

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