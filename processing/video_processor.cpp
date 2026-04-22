#include "video_processor.hpp"

double
VideoProcessor::getFps() const
{
    return fps;
}

int
VideoProcessor::getWidth() const
{
    return width;
}

int
VideoProcessor::getHeight() const
{
    return height;
}

VideoProcessor::VideoProcessor(std::filesystem::path input, std::filesystem::path output)
{
    // initialize capture & public properties
    videoCapture.open(input.string());
    if (!videoCapture.isOpened())
    {
        throw std::runtime_error("Video capture failed to open. Does the input file provided exist?");
    }
    fps = videoCapture.get(cv::CAP_PROP_FPS);
    width = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

    // initialize writer
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    videoWriter.open(output.string(), fourcc, fps, cv::Size(width, height));
}

VideoProcessor::~VideoProcessor()
{
    if (!captureFinished())
    {
        releaseCapture();
    }
    if (!writerFinished())
    {
        releaseWriter();
    }
}

// === READER METHODS ===

bool
VideoProcessor::getNextFrame(cv::Mat &output, bool grayscale)
{
    cv::Mat frame;
    if (!videoCapture.read(frame) || frame.empty())
    {
        releaseCapture();
        return false;
    }

    if (grayscale)
    {
        // Check if the frame actually needs conversion
        if (frame.channels() == 1)
        {
            output = frame.clone();
        }
        else
        {
            cv::cvtColor(frame, output, (frame.channels() == 3) ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
        }
    }
    else
    {
        output = frame.clone();
    }
    return true;
}

void
VideoProcessor::releaseCapture()
{
    videoCapture.release();
}

bool
VideoProcessor::captureFinished()
{
    return !videoCapture.isOpened();
}

// === WRITER METHODS ===

void
VideoProcessor::writeFrame(const cv::Mat &frame)
{
    cv::Mat buffer;
    if (frame.depth() != CV_8U)
    {
        cv::normalize(frame, buffer, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    else
    {
        buffer = frame;
    }

    if (videoWriter.isOpened() && !frame.empty())
    {
        videoWriter.write(buffer);
    }
}

void
VideoProcessor::releaseWriter()
{
    videoWriter.release();
}

bool
VideoProcessor::writerFinished()
{
    return !videoWriter.isOpened();
}
