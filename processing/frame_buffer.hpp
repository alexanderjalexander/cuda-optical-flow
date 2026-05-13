#ifndef FRAME_BUFFER_H
#define FRAME_BUFFER_H

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <memory>

class FrameBuffer
{
    private:
        cv::Mat *frames; // circular buffer of frames, allocated either in pinned or pageable memory
        
        size_t capacity; // total number of frames that can be stored in the buffer at once
        size_t size; // current number of frames in the buffer
        size_t front; // index of the oldest frame
        size_t back; // index of the next frame to be written

        bool usePinnedMemory;
        std::unique_ptr<cv::VideoCapture> source;

        int loadFrame();

    public:
        int initialize(std::unique_ptr<cv::VideoCapture> &src, size_t _capacity, bool _usePinnedMemory);
        bool empty() const;
        bool full() const;
        cv::Mat next();
        int release();
};

#endif