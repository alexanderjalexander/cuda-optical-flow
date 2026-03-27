#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include "lucasKanade.hpp"

using namespace cv;
using namespace std;

void
sparseLucasKanadeCPU(VideoInfo &video)
{
    bool drawContinuous = true;
    if (video.frames.empty()) return;

    Mat old_frame = video.frames[0];
    Mat mask = Mat::zeros(old_frame.size(), CV_8UC3);
    RNG rng;

    vector<Point2f> p0, p1;

    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarris = true;
    double k = 0.04;
    goodFeaturesToTrack(old_frame, p0, MAX_FEATURES, qualityLevel, minDistance, Mat(), blockSize, useHarris, k);

    int initialFeatures = p0.size();

    vector<Scalar> pt_colors;
    for (size_t i = 0; i < p0.size(); i++) {
        pt_colors.push_back(Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)));
    }

    for (size_t i = 1; i < video.frames.size(); i++)
    {
        Mat frame = video.frames[i];
        Mat output;
        cvtColor(frame, output, COLOR_GRAY2BGR);

        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_frame, frame, p0, p1, status, err, Size(15, 15), 2, criteria);

        vector<Point2f> good_new;

        // draw the points if the status is good.
        for (uint j = 0; j < p0.size(); j++)
        {
            if (status[j] == 1)
            {
                good_new.push_back(p1[j]);

                if (drawContinuous) {
                    line(mask, p1[j], p0[j], pt_colors[j], 2);
                    circle(output, p1[j], 5, pt_colors[j], -1);
                } else {
                    arrowedLine(output, p0[j], p1[j], pt_colors[j], 5, cv::LineTypes::LINE_AA, 0, .3);
                }
            }
        }

        // If we're drawing continuous lines throughout the whole thing.
        if (drawContinuous) {
            bitwise_or(output, mask, output);
        }

        video.outputFrames.push_back(output);
        old_frame = frame.clone();

        p0 = good_new;

        // Point replenishment
        if (p0.size() < initialFeatures * 0.7)
        {
            pt_colors.resize(p0.size());

            Mat exclusionMask = Mat::ones(old_frame.size(), CV_8U) * 255;
            for (int i = 0; i < p0.size(); i++) {
                circle(exclusionMask, p0[i], (int)minDistance, 0, -1);
            }

            vector<Point2f> newPoints;
            goodFeaturesToTrack(old_frame, newPoints, MAX_FEATURES, qualityLevel, minDistance, Mat(), blockSize, useHarris, k);

            for (int i = 0; i < newPoints.size(); i++)
            {
                p0.push_back(newPoints[i]);
                pt_colors.push_back(Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)));
            }
        }
    }
}
