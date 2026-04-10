#include "../processing/drawing.hpp"

#include "lucasKanade.hpp"

#include <iostream>

using namespace cv;
using namespace std;

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
        TermCriteria criteria =
            TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), LK_ITERATIONS, LK_EPSILON);
        calcOpticalFlowPyrLK(old_frame, frame, p0, p1, status, err, Size(15, 15), 2, criteria);

        drawOpticalFlow(output, mask, p0, p1, status, pt_colors, DRAW_CONTINUOUS_LINES);

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
