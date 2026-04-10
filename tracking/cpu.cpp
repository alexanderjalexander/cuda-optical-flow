#include "../processing/drawing.hpp"

#include "lucasKanade.hpp"

#include <iostream>

using namespace cv;
using namespace std;

/**
 * Host code that initiates a full flow of Sparse Lucas Kanade on the CPU.
 *
 * At first, this will calculate possibly good features using OpenCV's
 * goodFeaturesToTrack, using a Harris detector. Then, it will perform Lucas
 * Kanade flow calculations per each pair of frames, draw the flow result on the
 * result frames, and then continue until there are no more frames left to calculate
 * Lucas Kanade optical flow on.
 *
 * @param video the VideoInfo struct storing the initial frame's videos.
 */
void
sparseLucasKanadeCPU(VideoInfo &video)
{
    if (video.frames.empty())
    {
        return;
    }

    Mat old_frame = video.frames[0];
    Mat mask = Mat::zeros(old_frame.size(), CV_8UC3);
    RNG rng;

    vector<Point2f> p0, p1;

    int blockSize = 3;
    bool useHarris = true;
    goodFeaturesToTrack(old_frame, p0, MAX_FEATURES, QUALITY_LEVEL, HARRIS_DISTANCE, Mat(), blockSize, useHarris,
                        HARRIS_EPSILON);

    int initialFeatures = p0.size();

    vector<Scalar> pt_colors = getRandomColors(initialFeatures);

    for (size_t i = 1; i < video.frames.size(); i++)
    {
        Mat frame = video.frames[i];
        Mat output;
        cvtColor(frame, output, COLOR_GRAY2BGR);

        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), LK_ITERATIONS, LK_EPSILON);
        calcOpticalFlowPyrLK(old_frame, frame, p0, p1, status, err, Size(15, 15), 2, criteria);

        drawSparseOpticalFlow(output, mask, p0, p1, status, pt_colors, DRAW_CONTINUOUS_LINES);

        video.outputFrames.push_back(output);
        old_frame = frame.clone();

        // Frame Recalculation Logic
        vector<Point2f> good_new;
        for (uint j = 0; j < p0.size(); j++)
        {
            if (status[j] == 1)
            {
                good_new.push_back(p1[j]);
            }
        }

        p0 = good_new;

        // Point replenishment
        // if (p0.size() < initialFeatures * 0.7)
        // {
        //     pt_colors.resize(p0.size());

        //     Mat exclusionMask = Mat::ones(old_frame.size(), CV_8U) * 255;
        //     for (int i = 0; i < p0.size(); i++) {
        //         circle(exclusionMask, p0[i], (int)minDistance, 0, -1);
        //     }

        //     vector<Point2f> newPoints;
        //     goodFeaturesToTrack(old_frame, newPoints, MAX_FEATURES, qualityLevel, minDistance, Mat(), blockSize,
        //     useHarris, k);

        //     for (int i = 0; i < newPoints.size(); i++)
        //     {
        //         p0.push_back(newPoints[i]);
        //         pt_colors.push_back(Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)));
        //     }
        // }
    }
}

/**
 * Host code that initiates a full flow of Sparse Lucas Kanade on the CPU.
 *
 * At first, this will calculate possibly good features using OpenCV's
 * goodFeaturesToTrack, using a Harris detector. Then, it will perform Lucas
 * Kanade flow calculations per each pair of frames, draw the flow result on the
 * result frames, and then continue until there are no more frames left to calculate
 * Lucas Kanade optical flow on.
 *
 * @param video the VideoInfo struct storing the initial frame's videos.
 */
void
denseLucasKanadeCPU(VideoInfo &video)
{
    if (video.frames.empty())
    {
        return;
    }

    Mat old_frame = video.frames[0];
    Mat mask = Mat::zeros(old_frame.size(), CV_8UC3);
    RNG rng;

    int blockSize = 3;
    bool useHarris = true;

    vector<Point2f> p0, p1;

    p0.reserve(video.width * video.height);
    p1.reserve(video.width * video.height);

    for (int y = 0; y < video.height; y++)
    {
        for (int x = 0; x < video.width; x++)
        {
            p0.push_back(Point2f((float)x, (float)y));
        }
    }

    int initialFeatures = p0.size();

    vector<Scalar> pt_colors = getRandomColors(initialFeatures);

    for (size_t i = 1; i < video.frames.size(); i++)
    {
        Mat frame = video.frames[i];
        cv::Mat output = cv::Mat::zeros(video.width, video.height, CV_8UC3);

        // cvtColor(frame, output, COLOR_GRAY2BGR);

        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), LK_ITERATIONS, LK_EPSILON);
        calcOpticalFlowPyrLK(old_frame, frame, p0, p1, status, err, Size(15, 15), 2, criteria);

        drawDenseOpticalFlow(output, p0, p1, status);

        video.outputFrames.push_back(output);
        old_frame = frame.clone();

        // Frame Recalculation Logic
        vector<Point2f> good_new;
        for (int y = 0; y < video.height; y++)
        {
            for (int x = 0; x < video.width; x++)
            {
                p0[y * video.width + x].x = x;
                p0[y * video.width + x].y = y;
            }
        }

        p0 = good_new;
    }
}
