#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <linux/limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ __forceinline__ void
test_kernel()
{
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

    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video file." << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Video file opened successfully!" << std::endl;
    }

    // Read the first frame to confirm reading
    cv::Mat frame;
    bool ret = cap.read(frame);

    if (ret)
    {

        // Display the frame using imshow
        cv::imshow("First Frame", frame);
        cv::waitKey(0);          // Wait for a key press to close the window
        cv::destroyAllWindows(); // Close the window
    }
    else
    {
        std::cout << "Error: Could not read the frame." << std::endl;
    }

    // Release the video capture object
    cap.release();

    printf("Hello world!\n");
    return EXIT_SUCCESS;
}