#include "./lucasKanade.hpp"

#include <iostream>
#include <vector>

using namespace std;

#define MAGNITUDE_THRESHOLD 1.0f
#define ANGLE_THRESHOLD 0.1f

void
compareCPUSingleFlowOnDense(cv::Mat oldFrame, cv::Mat frame, vector<cv::Point2f> oldPoints, vector<cv::Point2f> points,
                            vector<uchar> status, ProgramFlags flags)
{
    cv::Mat output, outputParts[2];
    cv::calcOpticalFlowFarneback(oldFrame, frame, output, 0.5, MAX_PYR_LEVELS, LK_WINDOW_WIDTH, LK_ITERATIONS, 5, 1.1,
                                 cv::OPTFLOW_FARNEBACK_GAUSSIAN);
    cv::split(output, outputParts);

    cv::Mat magnitudeRef, angleRef;
    cartToPolar(outputParts[0], outputParts[1], magnitudeRef, angleRef, false);

    correctFlows = totalFlows = 0;

    for (size_t i = 0; i < oldPoints.size(); i++)
    {
        if (status[i] != 1)
        {
            continue;
        }
        float magnitude = sqrt(pow(points[i].x - oldPoints[i].x, 2) + pow(points[i].y - oldPoints[i].y, 2));
        float angle = atan2((points[i].y - oldPoints[i].y), (points[i].x - oldPoints[i].x));

        totalFlows++;
        if (std::abs(magnitudeRef.at<float>((int)points[i].y, (int)points[i].x) - magnitude) > MAGNITUDE_THRESHOLD)
        {
            continue;
        }
        float angleDiff = std::abs(angleRef.at<float>((int)points[i].y, (int)points[i].x) - angle);
        angleDiff = std::min(angleDiff, (float)(2 * M_PI) - angleDiff);
        if (angleDiff > ANGLE_THRESHOLD)
        {
            continue;
        }

        correctFlows++;
    }
}

void
compareGPUSingleFlowOnDense(cv::Mat oldFrame, cv::Mat frame, cv::Vec3f *oldPoints, cv::Vec3f *points, int featureCount,
                            ProgramFlags flags)
{
    cv::Mat output, outputParts[2];
    cv::calcOpticalFlowFarneback(oldFrame, frame, output, 0.5, MAX_PYR_LEVELS, LK_WINDOW_WIDTH, LK_ITERATIONS, 5, 1.1,
                                 cv::OPTFLOW_FARNEBACK_GAUSSIAN);
    cv::split(output, outputParts);

    cv::Mat magnitudeRef, angleRef;
    cartToPolar(outputParts[0], outputParts[1], magnitudeRef, angleRef, false);

    correctFlows = totalFlows = 0;

    for (int i = 0; i < featureCount; i++)
    {
        if (points[i][2] != 1)
        {
            continue;
        }
        float magnitude = sqrt(pow(points[i][0] - oldPoints[i][0], 2) + pow(points[i][1] - oldPoints[i][1], 2));
        float angle = atan2((points[i][1] - oldPoints[i][1]), (points[i][0] - oldPoints[i][0]));

        totalFlows++;
        if (std::abs(magnitudeRef.at<float>((int)points[i][1], (int)points[i][0]) - magnitude) > MAGNITUDE_THRESHOLD)
        {
            continue;
        }
        float angleDiff = std::abs(angleRef.at<float>((int)points[i][1], (int)points[i][0]) - angle);
        angleDiff = std::min(angleDiff, (float)(2 * M_PI) - angleDiff);
        if (angleDiff > ANGLE_THRESHOLD)
        {
            continue;
        }

        correctFlows++;
    }
}
