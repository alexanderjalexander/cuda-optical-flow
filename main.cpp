#include <opencv2/opencv.hpp>
#include <sys/stat.h>

#include "processing/video_io.hpp"
#include "timing/stopwatch.hpp"
#include "tracking/lucasKanade.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <linux/limits.h>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void
usage(const char *progname)
{
    printf("Usage: %s\n", progname);
}

int
main(int argc, char *argv[])
{
    // Reading the videos
    std::filesystem::path current_dir = std::filesystem::current_path();
    std::string file_input = "Slow-Traffic";
    std::filesystem::path full_path = current_dir / "inputs" / (file_input + ".mp4");
    std::filesystem::path sparseGpuOutputPath = current_dir / "outputs" / (file_input + "-SparseGPU.mp4");
    std::filesystem::path sparseCpuOutputPath = current_dir / "outputs" / (file_input + "-SparseCPU.mp4");

    VideoInfo sparseCpuVideo;
    VideoInfo sparseGpuVideo;
    startStopwatch();
    if (readVideo(sparseGpuVideo, full_path) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    stopStopwatch();
    if (copyVideo(sparseCpuVideo, sparseGpuVideo) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    // GPU Lucas Kanade

    std::cout << std::endl << "Starting GPU Lucas Kanade..." << std::endl;
    std::cout << "Frames to Process: " << sparseGpuVideo.frames.size() << std::endl;
    startStopwatch();
    sparseLucasKanadeGPU(sparseGpuVideo);
    stopStopwatch();

    std::cout << std::endl << "Writing GPU Lucas Kanade output to video..." << std::endl;

    startStopwatch();
    if (writeVideo(sparseGpuVideo, sparseGpuOutputPath) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    stopStopwatch();

    // CPU Lucas Kanade

    std::cout << std::endl << "Starting CPU Lucas Kanade..." << std::endl;
    std::cout << "Frames to Process: " << sparseCpuVideo.frames.size() << std::endl;
    startStopwatch();
    sparseLucasKanadeCPU(sparseCpuVideo);
    stopStopwatch();

    std::cout << std::endl << "Writing CPU Lucas Kanade output to video..." << std::endl;

    startStopwatch();
    if (writeVideo(sparseCpuVideo, sparseCpuOutputPath) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    stopStopwatch();

    return EXIT_SUCCESS;
}
