#include "video_io.hpp"

int
readVideo(VideoInfo &video, std::filesystem::path video_path)
{
    cv::VideoCapture cap(video_path.string());

    video.totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
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
        std::cout << "Total frames in video: " << video.totalFrames << std::endl;
        std::cout << "Video FPS: " << video.fps << std::endl;
    }

    cv::Mat frame, gray;
    video.frames.reserve(video.totalFrames > 0 ? video.totalFrames : 100);

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

int
copyVideo(VideoInfo &dstVideo, VideoInfo &srcVideo)
{
    dstVideo.fps = srcVideo.fps;
    dstVideo.totalFrames = srcVideo.totalFrames;
    dstVideo.width = srcVideo.width;
    dstVideo.height = srcVideo.height;
    for (int i = 0; i < srcVideo.frames.size(); i++)
    {
        dstVideo.frames.push_back(srcVideo.frames[i].clone());
    }
    return EXIT_SUCCESS;
}
