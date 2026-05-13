#include "frame_buffer.hpp"

#include <cstdlib>
#include <iostream>
#include <utility>


int FrameBuffer::initialize(std::unique_ptr<cv::VideoCapture> &source, size_t _capacity, bool _usePinnedMemory)
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
    
    this->usePinnedMemory = _usePinnedMemory;
    this->frames = new cv::Mat[capacity]{};
    // FIXME need to set up the frames to use pre-allocated pinned memory based on the frame size, then copy into this buffer after reading from the source
    if (usePinnedMemory) {
        for (size_t i = 0; i < capacity; ++i)
        {
            cudaHostRegister(frames[i].data, frames[i].total() * frames[i].elemSize(), cudaHostRegisterDefault);
        }
    }

    this->source = std::move(source);
    if (!this->source || !this->source->isOpened())
    {
        std::cerr << "Error: FrameBuffer source is not open!" << std::endl;
        return EXIT_FAILURE;
    }

    // load initial frames to fill the buffer
    for (size_t i = 0; i < _capacity + 1; ++i)
    {
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

// Return the next frame from the front of the buffer, and load the next frame from the source video into the back of the buffer.\
// If the buffer is empty, return an empty Mat.
cv::Mat FrameBuffer::next()
{
    if (empty())
    {
        // std::cout << "FrameBuffer is empty!" << std::endl;
        return cv::Mat();
    }

    cv::Mat frame = frames[front];
    front = (front + 1) % capacity;
    size--;

    loadFrame();

    return frame;
}

int FrameBuffer::loadFrame()
{
    if (!frames)
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

    // todo: what to do when there are no more frames to read?
    cv::Mat frame;
    if (this->source->read(frame) && !frame.empty())
    {
        frames[back] = frame;
        back = (back + 1) % capacity;
    }

    return EXIT_SUCCESS;
}

int FrameBuffer::release()
{
    delete[] frames;
    frames = nullptr;
    if (source)
    {
        source->release();
    }
    return EXIT_SUCCESS;
}