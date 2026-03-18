#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <linux/limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "gpu/lk.cuh"

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

    cv::VideoCapture cap(full_path.string());

    double totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double videoFPS = cap.get(cv::CAP_PROP_FPS);

    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video file." << std::endl;
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Video file opened successfully!" << std::endl;
        std::cout << "Total frames in video: " << totalFrames << std::endl;
        std::cout << "Video FPS: " << videoFPS << std::endl;
    }

    // CPU memory frame buffer
    cv::Mat frame, gray;
    std::vector<cv::Mat> frames;
    frames.reserve(totalFrames > 0 ? totalFrames : 1000);

    while (cap.read(frame))
    {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        frames.push_back(gray.clone());
    }
    std::cout << "Buffered " << frames.size() << " frames." << std::endl;

    // GPU processing: compute spatial derivatives for all frames
    std::cout << "\nSending frames to GPU...\n";

    if (!frames.empty())
    {
        int width = frames[0].cols;
        int height = frames[0].rows;
        size_t framePixels = width * height;

        for (size_t i = 0; i < frames.size(); ++i)
        {
            // TODO: Flesh this out
            processFrameOnGPU(frames[i]);

            if ((i + 1) % 50 == 0 || i == 0)
            {
                std::cout << "Frames Processed: " << (i + 1) << " / " << frames.size() << std::endl;
            }
        }

        std::cout << "GPU processing complete! Processed all " << frames.size() << " frames.\n";
    }

    // Release the video capture object
    cap.release();
    return EXIT_SUCCESS;
}