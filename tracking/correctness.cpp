#include "./lucasKanade.hpp"

#include <iostream>
#include <vector>

#define RELATIVE_MAGNITUDE_THRESHOLD 0.15f
#define MAGNITUDE_THRESHOLD 2.0f
#define ANGLE_THRESHOLD 5.0f

uint64 correctFlows = 0;
uint64 totalFlows = 0;
uint64 angleOffFlows = 0;
uint64 magOffFlows = 0;
uint64 pointBadFlows = 0;

void
compareGPULucasKanadeFlow(cv::Mat oldFrame, cv::Mat frame, cv::Vec3f *oldPoints, cv::Vec3f *points, int featureCount,
                            ProgramFlags flags)
{
    std::vector<cv::Point2f> p0_cpu, p1_cpu;
    std::vector<int> idxMap;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), LK_ITERATIONS, LK_EPSILON);

    for (int i = 0; i < featureCount; i++)
    {
        if (points[i][2] != 1)
        {
            continue;
        }
        totalFlows++;
        p0_cpu.push_back(cv::Point2f(oldPoints[i][0], oldPoints[i][1]));
        idxMap.push_back(i);
    }

    if (p0_cpu.empty())
    {
        return;
    }

    cv::calcOpticalFlowPyrLK(oldFrame, frame, p0_cpu, p1_cpu, status, err, cv::Size(LK_WINDOW_WIDTH, LK_WINDOW_WIDTH),
                             (flags.mipMap ? MAX_PYR_LEVELS - 1 : 0), criteria);

    for (int i = 0; i < idxMap.size(); i++)
    {
        int idx = idxMap[i];

        // Bad track, point doesn't exist.
        if (status[i] != 1)
        {
            pointBadFlows++;
            continue;
        }

        // displacement from GPU LK
        float gpuDx = points[idx][0] - oldPoints[idx][0];
        float gpuDy = points[idx][1] - oldPoints[idx][1];
        float gpuMag = std::hypot(gpuDx, gpuDy);

        // displacement from CPU LK
        float cpuDx = p1_cpu[i].x - p0_cpu[i].x;
        float cpuDy = p1_cpu[i].y - p0_cpu[i].y;
        float cpuMag = std::hypot(cpuDx, cpuDy);

        // magnitude comparison
        float magErr = std::fabs(gpuMag - cpuMag);
        // also do relative magnitude...
        // if it's like 45 px to 48 px, that's not relatively off.
        float magRelErr = magErr / (cpuMag + 1e-3f);

        if (magErr > MAGNITUDE_THRESHOLD && magRelErr > RELATIVE_MAGNITUDE_THRESHOLD)
        {
            magOffFlows++;
            continue;
        }

        // angle comparison
        if (cpuMag > 0.5f && gpuMag > 0.5f)
        {
            // inverse tan for Theta, then get the absolute values
            float angleDiff = std::fabs(std::atan2(gpuDy, gpuDx) - std::atan2(cpuDy, cpuDx));
            // wrap it from 0 to pi, get rid of negatives
            if (angleDiff > (float)M_PI)
            {
                angleDiff = 2.0f * (float)M_PI - angleDiff;
            }
            // final angle comparison
            if (angleDiff * (180.0f / (float)M_PI) > ANGLE_THRESHOLD)
            {
                angleOffFlows++;
                continue;
            }
        }

        correctFlows++;
    }
}
