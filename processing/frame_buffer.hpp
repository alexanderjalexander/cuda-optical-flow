#ifndef FRAME_BUFFER_H
#define FRAME_BUFFER_H

#include <opencv2/opencv.hpp>
#include "opencv2/core/cuda.hpp"

#include <cstddef>
#include <memory>
#include <utility>

// FrameBuffer handles batched frame loading for the CPU and GPU by maintaining a circular buffer of frames.
// Whenever a frame is read from the front of the buffer, the next frame from the source video is loaded into the back of the buffer.
// By using GpuMat, the frames can be automatically transferred to GPU memory when they're loaded from the source video.
// Depending on the value of `useStreams`, this transfer can be done asynchronously to allow for overlapping of GPU data transfer and computation.
class FrameBuffer
{
    private:
        cv::cuda::GpuMat *frames; // circular buffer of frames, allocated either in pinned or pageable memory
        cv::Mat *cpuFrames; // corresponding CPU frames

        size_t capacity; // total number of frames that can be stored in the buffer at once
        size_t size; // current number of frames in the buffer
        size_t front; // index of the oldest frame
        size_t back; // index of the next frame to be written

        std::unique_ptr<cv::VideoCapture> source;

        bool useStreams; // whether to use CUDA streams for asynchronous transfers
        cv::cuda::Stream stream; // CUDA stream for asynchronous transfers

        int loadFrame();

    public:
        int initialize(std::unique_ptr<cv::VideoCapture> &src, size_t _capacity, bool _useStreams);
        bool empty() const;
        bool full() const;
        std::pair<cv::cuda::GpuMat, cv::Mat> next(bool ignoreGpu = false, bool ignoreCpu = false);
        int release();
        void syncStream() { stream.waitForCompletion(); }
        void resetCapture();
};

#endif