#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <linux/limits.h>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "tracking/lucasKanade.hpp"
#include "timing/stopwatch.hpp"
#include "processing/video_io.hpp"

int
main(int argc, char *argv[])
{
    // Reading the videos

    std::filesystem::path current_dir = std::filesystem::current_path();
    std::string file_input = "Slow-Traffic";
    std::filesystem::path full_path = current_dir / "inputs" / (file_input + ".mp4");
    std::filesystem::path gpu_output_path = current_dir / "outputs" / (file_input + "-GPU.mp4");
    std::filesystem::path cpu_output_path = current_dir / "outputs" / (file_input + "-CPU.mp4");

    VideoInfo cpuVideo;
    VideoInfo gpuVideo;
    startStopwatch();
    if (readVideo(gpuVideo, full_path) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    stopStopwatch();
    if (copyVideo(cpuVideo, gpuVideo) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    // GPU Lucas Kanade

    std::cout << std::endl << "Starting GPU Lucas Kanade..." << std::endl;
    std::cout << "Frames to Process: " << gpuVideo.frames.size() << std::endl;
    startStopwatch();
    lucasKanadeGPU(gpuVideo);
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
    lucasKanadeCPU(cpuVideo);
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