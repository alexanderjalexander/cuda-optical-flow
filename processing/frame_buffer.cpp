#include "frame_buffer.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/mat.hpp"

#include <cstdlib>
#include <iostream>
#include <utility>

// Start at 1, so that we don't accidentally overwrite data being processed on the GPU.
#define FRONT_START 1

int FrameBuffer::initialize(std::unique_ptr<cv::VideoCapture> &source, size_t _capacity, bool _useStreams)
{
    if (_capacity == 0)
    {
        std::cerr << "Error: FrameBuffer capacity must be greater than zero!" << std::endl;
        return EXIT_FAILURE;
    }

    this->capacity = _capacity;
    this->size = 0;
    this->front = FRONT_START;
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
        // printf("Loading frame %d...\n", i);
        frames[i % capacity] = cv::cuda::GpuMat();
        if (i < (capacity - FRONT_START))
        {
            if (loadFrame() != EXIT_SUCCESS)
            {
                std::cerr << "Error: Failed to load frame " << i << " into FrameBuffer!" << std::endl;
                return EXIT_FAILURE;
            }
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

    // Ensure the front frame is fully uploaded before we hand it out
    if (useStreams)
    {
        stream.waitForCompletion();
    }

    // return the front frame
    cv::cuda::GpuMat gpuFrame = ignoreGpu ? cv::cuda::GpuMat() : frames[front];
    cv::Mat cpuFrame = ignoreCpu ? cv::Mat() : cpuFrames[front];
    front = (front + 1) % capacity;
    // printf("Front updated to = %d\n", front);
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
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // transfer frame to GPU memory
        if (useStreams)
        {
            // printf("Back = %d\n", back);
            frames[back].upload(gray, stream); // transfer frame to GPU asynchronously
        }
        else
        {
            // printf("Back = %d\n", back);
            frames[back].upload(gray); // transfer frame to GPU synchronously
        }
        cpuFrames[back] = gray; // keep a copy of the original frame on the CPU as well
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

void FrameBuffer::resetCapture()
{
    if (source && source->isOpened())
    {
        source->set(cv::CAP_PROP_POS_FRAMES, 0);
        this->front = FRONT_START;
        this->back = 0;
        this->size = 0;

        // refill the buffer like initialize() does
        for (size_t i = 0; i < capacity - FRONT_START; ++i)
        {
            if (loadFrame() != EXIT_SUCCESS)
            {
                std::cerr << "Error: Failed to refill FrameBuffer during reset!" << std::endl;
                return;
            }
        }

        if (useStreams)
        {
            stream.waitForCompletion();
        }
    }
}