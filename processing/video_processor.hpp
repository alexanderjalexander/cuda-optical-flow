#ifndef VIDEO_PROCESSOR_HPP
#define VIDEO_PROCESSOR_HPP

#include <opencv2/opencv.hpp>

#include <filesystem>

/**
 * Non-copyable class that manages video I/O for a single video.
 *
 * Wraps around a cv::VideoCapture reader and cv::VideoWriter writer, tracking FPS and frame size.
 * Provides methods to read/write frames, check completion, and release resources.
 *
 * Makes the faithful assumption that you're performing modifications on the video, hence the retention of
 * video FPS, width, and height.
 */
class VideoProcessor
{
private:
    // The main video reader.
    cv::VideoCapture videoCapture;
    // The main video writer.
    cv::VideoWriter videoWriter;

    // The video's FPS.
    double fps;
    // The video's width.
    int width;
    // The video's height.
    int height;

    /**
     * ABANDONED_TODO: Reader Thread
     * - Queue of frames that have been read
     * - Mutex for queue
     * - The thread itself
     * - Atomic bool for whether the capture is done or not
     * ABANDONED_TODO: Writer Thread
     * - Queue of frames to write to disk
     * - Mutex for queue
     * - The thread itself
     * - Atomic bool for whether the write is done or not
     */

public:
    /**
     * Getter for the video's FPS.
     *
     * @returns The video's FPS.
     */
    double getFps() const;

    /**
     * Getter for the video's width.
     *
     * @returns The video's width.
     */
    int getWidth() const;

    /**
     * Getter for the video's height.
     *
     * @returns The video's height.
     */
    int getHeight() const;

    /**
     * Constructs a video processor given an existing video
     *
     * @param input The filesystem path according to the input video to read from. Must exist.
     * @param output The filesystem path according to the output video to write to.
     *
     * @returns a new VideoProcessor class.
     */
    VideoProcessor(std::filesystem::path input, std::filesystem::path output);

    /**
     * Destroys a VideoProcessor.
     */
    ~VideoProcessor();

    // these we won't need, below, so goodbye

    VideoProcessor(const VideoProcessor &) = delete;
    VideoProcessor &operator=(const VideoProcessor &) = delete;

    /**
     * Gets the next frame from an
     *
     * @param output The buffer to write the next frame to.
     * @param grayscale Whether or not we want to convert to grayscale. Defaults to true.
     *
     * @returns T/F depending on whether we got the next frame successfully.
     */
    bool getNextFrame(cv::Mat &output, bool grayscale = true);

    /**
     * Releases the capture of the video.
     */
    void releaseCapture();

    /**
     * Tells whether or not the capture is in progress or done.
     *
     * @returns True if the capture is done/gone, false if the capture is still active.
     */
    bool captureFinished();

    /**
     * Writes a frame to the desired output path, obtained in the constructor.
     */
    void writeFrame(const cv::Mat &frame);

    /**
     * Closes the writer, finishing the video-writing process.
     */
    void releaseWriter();

    /**
     * Tells whether or not the writer is in progress or done.
     *
     * @returns True if the writer is done/gone, false if the writer is still active.
     */
    bool writerFinished();
};

#endif
