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

static bool
cudaDeviceAvailable()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0)
    {
        std::cerr << "No CUDA device available for GPU computation!" << std::endl;
        return false;
    }
    return true;
}

static bool
fileAtPathExists(char *path)
{
    std::string strPath = path;
    std::filesystem::path absPath = std::filesystem::absolute(strPath);
    return std::filesystem::exists(absPath);
}

static std::filesystem::path
returnCanonicalFilePath(std::string path)
{
    std::string strPath = path;
    return std::filesystem::canonical(strPath);
}

static std::filesystem::path
returnOutputFilePath(std::filesystem::path path, std::string suffix)
{
    std::filesystem::path outputDir = std::filesystem::current_path() / "outputs";
    std::filesystem::create_directories(outputDir);
    std::string outputFileName = path.stem().string() + suffix + ".mp4";
    std::filesystem::path outputPath = outputDir / outputFileName;
    return outputPath;
}

const char flagChars[] = "st";

static void
usage(const char *progname)
{
    fprintf(stderr, "Usage: %s [-%s] videoName\n", progname, flagChars);
    fprintf(stderr, "\t-s --> Run in 'Statistics Mode'\n");
    fprintf(stderr, "\t-s --> Run with Texture Memory\n");
    fprintf(stderr, "*NOTE* videoName is the relative path to a file, extension included.\n");
}

struct ProgramFlags
{
    bool statsMode;
    bool textureMem;
};

int
main(int argc, char *argv[])
{
    // Check if a GPU exists before running anything
    if (!cudaDeviceAvailable())
    {
        std::cerr << "No CUDA device available for GPU computation!" << std::endl;
        return EXIT_FAILURE;
    }

    // Various options flags
    ProgramFlags progFlags = {0};

    // Other misc stuff
    char *progname = argv[0];

    // Parse all options
    int opt;
    while ((opt = getopt(argc, argv, flagChars)) != -1)
    {
        switch (opt)
        {
        case 's':
            std::cout << "Statistics Mode enabled." << std::endl;
            progFlags.statsMode = true;
            break;
        case 't':
            std::cout << "Texture Memory enabled." << std::endl;
            progFlags.textureMem = true;
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
    // TODO - maybe allow the user to specify the full path to the input/output video to support other file formats
    if (!fileAtPathExists(fileInputPath))
    {
        std::cerr << "File at '" << fileInputPath << "' does not exist." << std::endl;
        return EX_NOINPUT;
    }
    std::filesystem::path fullFileInputPath = returnCanonicalFilePath(fileInputPath);

    VideoInfo video;
    startStopwatch();
    if (readVideo(video, fullFileInputPath) != EXIT_SUCCESS)
    {
        return EX_NOINPUT;
    }
    stopStopwatch();

    if (progFlags.statsMode)
    {
        // Run both algorithms in statistics mode
        int returnCode;
        // if ((returnCode = recordStatsSparseLucasKanade(true, progFlags.textureMem, video)) != EXIT_SUCCESS)
        // {
        //     return returnCode;
        // }
        if ((returnCode = recordStatsSparseLucasKanade(false, progFlags.textureMem, video)) != EXIT_SUCCESS)
        {
            return returnCode;
        }
    }
    else
    {
        // Run both algorithms normally, and output the result to a new vid in the /outputs folder
        std::filesystem::path sparseGpuOutputPath = returnOutputFilePath(fullFileInputPath, "-SparseGPU");
        std::filesystem::path sparseCpuOutputPath = returnOutputFilePath(fullFileInputPath, "-SparseCPU");

        // CPU Lucas Kanade
        std::cout << std::endl << "Starting CPU Lucas Kanade..." << std::endl;
        std::cout << "Frames to Process: " << video.frames.size() << std::endl;
        startStopwatch();
        sparseLucasKanadeCPU(video);
        stopStopwatch();

        std::cout << std::endl << "Writing CPU Lucas Kanade output to video..." << std::endl;
        startStopwatch();
        if (writeVideo(video, sparseCpuOutputPath) != EXIT_SUCCESS)
        {
            return EXIT_FAILURE;
        }
        stopStopwatch();

        // Super necessary for the sake of preventing excessive memory-hogging
        video.outputFrames.clear();

        // GPU Lucas Kanade
        std::cout << std::endl << "Starting GPU Lucas Kanade..." << std::endl;
        std::cout << "Frames to Process: " << video.frames.size() << std::endl;
        startStopwatch();
        if (progFlags.textureMem)
        {
            sparseLucasKanadeGPUTex(video);
        }
        else
        {
            sparseLucasKanadeGPU(video);
        }
        stopStopwatch();

        std::cout << std::endl << "Writing GPU Lucas Kanade output to video..." << std::endl;
        startStopwatch();
        if (writeVideo(video, sparseGpuOutputPath) != EXIT_SUCCESS)
        {
            return EXIT_FAILURE;
        }
        stopStopwatch();
    }

    return EXIT_SUCCESS;
}
