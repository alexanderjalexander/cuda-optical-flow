#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <linux/limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

__global__ void
test_kernel()
{
    printf("Hello world!");
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

    size_t totalBytes = 0;
    while (cap.read(frame))
    {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        frames.push_back(gray.clone());
        totalBytes += gray.total() * gray.elemSize();
    }
    std::cout << "Buffered " << frames.size() << " frames." << std::endl;
    std::cout << "Approx bytes in frame buffers: " << totalBytes << "\n";

    // Release the video capture object
    cap.release();

    printf("Finished reading video!\n");
    return EXIT_SUCCESS;
}