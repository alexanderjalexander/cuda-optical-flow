#include "frame_buffer.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/mat.hpp"

#include <cstdlib>
#include <iostream>
#include <utility>

int FrameBuffer::initialize(std::unique_ptr<cv::VideoCapture> &source, size_t _capacity, bool _useStreams)
{
    if (_capacity == 0)
    {
        std::cerr << "Error: FrameBuffer capacity must be greater than zero!" << std::endl;
        return EXIT_FAILURE;
    }
    
    this->capacity = _capacity;
    this->size = 0;
    this->front = 0;
    this->back = 0;
    this->useStreams = _useStreams;
    this->frames = new cv::cuda::GpuMat[capacity]();
    this->cpuFrames = new cv::Mat[capacity]();

    this->source = std::move(source);
    if (!this->source || !this->source->isOpened())
    {
        std::cerr << "Error: FrameBuffer source is not open!" << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate GpuMats here to avoid repeatedly re-initializing on the GPU,
    // and simultaneously fill initial frames with data
    for (size_t i = 0; i < _capacity; ++i)
    {
        frames[i] = cv::cuda::GpuMat();
        if (loadFrame() != EXIT_SUCCESS)
        {
            std::cerr << "Error: Failed to load frame " << i << " into FrameBuffer!" << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

bool FrameBuffer::empty() const
{
    return size == 0;
}

bool FrameBuffer::full() const
{
    return size == capacity;
}

// Return the next frame from the front of the buffer, and load the next frame from the source video into the back of the buffer.
// Return return a pair containing the GpuMat and corresponding Mat, or return empty for either type depending on the values of `ignoreGpu` and `ignoreCpu`.
// If the buffer is empty, return a pair of empty data.
std::pair<cv::cuda::GpuMat, cv::Mat> FrameBuffer::next(bool ignoreGpu, bool ignoreCpu)
{
    if (empty())
    {
        // std::cout << "FrameBuffer is empty!" << std::endl;
        return std::make_pair(cv::cuda::GpuMat(), cv::Mat()); // return empty pair
    }

    // return the front frame
    cv::cuda::GpuMat gpuFrame = ignoreGpu ? cv::cuda::GpuMat() : frames[front];
    cv::Mat cpuFrame = ignoreCpu ? cv::Mat() : cpuFrames[front];
    front = (front + 1) % capacity;
    size--;

    // add a new frame to the back of the frame
    loadFrame();

    return std::make_pair(gpuFrame, cpuFrame);
}

int FrameBuffer::loadFrame()
{
    if (!frames || !cpuFrames)
    {
        std::cerr << "Error: FrameBuffer frames not initialized!" << std::endl;
        return EXIT_FAILURE;
    }

    if (full())
    {
        return EXIT_SUCCESS;
    }

    if (!this->source || !this->source->isOpened())
    {
        std::cerr << "Error: FrameBuffer source is not open!" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat frame;
    // copy the data to the GPU if there are any frames remaining
    if (this->source->read(frame) && !frame.empty())
    {
        // transfer frame to GPU memory
        if (useStreams)
        {
            frames[back].upload(frame, stream); // transfer frame to GPU asynchronously
        }
        else
        {
            frames[back].upload(frame); // transfer frame to GPU synchronously
        }
        cpuFrames[back] = frame; // keep a copy of the original frame on the CPU as well
        back = (back + 1) % capacity;
        size++;
    }
    // if there are no more frames, still return success but don't modify any data

    return EXIT_SUCCESS;
}

int FrameBuffer::release()
{
    delete[] frames;
    frames = nullptr;
    delete[] cpuFrames;
    cpuFrames = nullptr;

    if (source)
    {
        source->release();
    }
    
    this->capacity = 0;
    this->size = 0;
    this->front = 0;
    this->back = 0;
    return EXIT_SUCCESS;
}