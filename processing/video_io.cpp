#include "video_io.hpp"

/**
 * Reads a video from the filesystem and stores the video and relevant info
 * into a struct.
 *
 * Converts the video to grayscale.
 *
 * @param video the VideoInfo struct to store the initial video's frames.
 * @param video_path the path to the video on the filesystem.
 *
 * @return EXIT_SUCCESS or EXIT_FAILURE, depending on the result of the read.
 */
int
readVideo(VideoInfo &video, std::filesystem::path video_path)
{
    cv::VideoCapture cap(video_path.string());

    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    video.fps = cap.get(cv::CAP_PROP_FPS);

    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video file." << std::endl;
        cap.release();
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Video file opened successfully!" << std::endl;
        std::cout << "Total frames in video: " << totalFrames << std::endl;
        std::cout << "Video FPS: " << video.fps << std::endl;
    }

    cv::Mat frame, gray;
    video.frames.reserve(totalFrames > 0 ? totalFrames : 100);

    int i = 0;
    while (cap.read(frame))
    {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        video.frames.push_back(gray.clone());
        i++;
    }
    cap.release();
    std::cout << "Buffered " << video.frames.size() << " frames." << std::endl;

    return EXIT_SUCCESS;
}

/**
 * Reads a video from the corresponding struct and writes the output buffer frames
 * to an MP4 file on the system.
 *
 * @param video the VideoInfo struct storing the initial video's frames.
 * @param video_path the path to write the output video to on the filesystem.
 *
 * @return EXIT_SUCCESS or EXIT_FAILURE, depending on the result of the write.
 */
int
writeVideo(VideoInfo &video, std::filesystem::path video_path)
{
    if (video.outputFrames.empty())
    {
        std::cerr << "Error: No frames to write." << std::endl;
        return EXIT_FAILURE;
    }

    int width = video.outputFrames[0].cols;
    int height = video.outputFrames[0].rows;
    bool isColor = (video.outputFrames[0].channels() == 3);

    cv::VideoWriter writer(video_path.string(), cv::VideoWriter::fourcc('a', 'v', 'c', '1'), video.fps,
                           cv::Size(width, height), isColor);

    if (!writer.isOpened())
    {
        std::cerr << "Error: Could not open output video for writing: " << video_path << std::endl;
        writer.release();
        return EXIT_FAILURE;
    }

    cv::Mat writeFrame;
    for (size_t i = 0; i < video.outputFrames.size(); ++i)
    {
        if (video.outputFrames[i].depth() != CV_8U)
        {
            cv::normalize(video.outputFrames[i], writeFrame, 0, 255, cv::NORM_MINMAX, CV_8U);
        }
        else
        {
            writeFrame = video.outputFrames[i];
        }

        writer.write(writeFrame);
    }

    writer.release();
    std::cout << "Wrote " << video.outputFrames.size() << " frames to: " << video_path << std::endl;
    return EXIT_SUCCESS;
}

/**
 * Copies a video from one struct to the next.
 *
 * @param dstVideo the VideoInfo struct to copy to.
 * @param srcVideo the VideoInfo struct to copy from.
 *
 * @return EXIT_SUCCESS
 */
int
copyVideo(VideoInfo &dstVideo, VideoInfo &srcVideo)
{
    dstVideo.fps = srcVideo.fps;
    dstVideo.width = srcVideo.width;
    dstVideo.height = srcVideo.height;
    for (int i = 0; i < srcVideo.frames.size(); i++)
    {
        dstVideo.frames.push_back(srcVideo.frames[i].clone());
    }
    return EXIT_SUCCESS;
}
