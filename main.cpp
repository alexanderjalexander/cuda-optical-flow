#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

#include "processing/video_io.hpp"
#include "timing/statistics.hpp"
#include "timing/stopwatch.hpp"
#include "tracking/lucasKanade.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <linux/limits.h>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <sysexits.h>
#include <unistd.h>

const char flags[] = "s";

static void
usage(const char *progname)
{
    fprintf(stderr, "Usage: %s [-%s] videoName\n", progname, flags);
    fprintf(stderr, "\t-s --> Run in 'Statistics Mode'\n");
    fprintf(stderr, "*NOTE* videoName is an input file within the /inputs folder, and must not "
                    "contain the video extension. It is expected to be a .mp4 file.\n");
}

int
main(int argc, char *argv[])
{
    // Various options flags
    bool statsModeEnabled = false;

    // Other misc stuff
    char *progname = argv[0];

    // Parse all options
    int opt;
    while ((opt = getopt(argc, argv, flags)) != -1)
    {
        switch (opt)
        {
        case 's':
            std::cout << "Statistics Mode enabled." << std::endl;
            statsModeEnabled = true;
            break;
        default:
            usage(progname);
            return EX_USAGE;
        }
    }

    // Parse options after arguments
    argc -= optind;
    argv += optind;

    if (argc != 1)
    {
        usage(progname);
        return EX_USAGE;
    }

    char *fileInputPath = argv[0];

    // Reading the videos
    std::filesystem::path current_dir = std::filesystem::current_path();
    std::string file_input = fileInputPath;
    std::filesystem::path full_path = current_dir / "inputs" / (file_input + ".mp4");
    // TODO - maybe allow the user to specify the full path to the input/output video to support other file formats

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

    if (!std::filesystem::exists(full_path))
    {
        std::cerr << "File at '" << full_path << "' does not exist." << std::endl;
        return EX_NOINPUT;
    }

    // Check if a GPU exists before running anything
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0)
    {
        std::cerr << "No CUDA device available for GPU computation!" << std::endl;
        return EXIT_FAILURE;
    }

    if (statsModeEnabled)
    {
        // Run both algorithms in statistics mode
        int returnCode;
        if ((returnCode = recordStatsSparseLucasKanade(true, sparseCpuVideo)) != EXIT_SUCCESS)
        {
            return returnCode;
        }
        if ((returnCode = recordStatsSparseLucasKanade(false, sparseGpuVideo)) != EXIT_SUCCESS)
        {
            return returnCode;
        }
    }
    else
    {
        // Run both algorithms normally, and output the result to a new vid in the /outputs folder
        std::filesystem::path sparseGpuOutputPath = current_dir / "outputs" / (file_input + "-SparseGPU.mp4");
        std::filesystem::path sparseCpuOutputPath = current_dir / "outputs" / (file_input + "-SparseCPU.mp4");

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
    }

    return EXIT_SUCCESS;
}
