#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

void
lucasKanadeCPU(const cv::Mat &prevFrame, const cv::Mat &frame, cv::Mat &result, int maxFeatures)
{
    // TODO: Completely refactor this
    // Logic needs to be changed, because feature detections are not persistent from the first frames.

    Mat prevFrameCopy;
    Mat frameCopy;
    vector<Point2f> p0, p1;

    // defensive guard... just in case
    if (prevFrame.channels() == 3)
        cvtColor(prevFrame, prevFrameCopy, COLOR_BGR2GRAY);
    else
        prevFrameCopy = prevFrame.clone();

    if (frame.channels() == 3)
        cvtColor(frame, frameCopy, COLOR_BGR2GRAY);
    else
        frameCopy = frame.clone();

    // Take first frame and find corners in it
    // old frame & old gray are prevFrame
    // frame and frame gray are frame
    double qualityLevel = 0.005;
    double minDistance = 3;
    goodFeaturesToTrack(prevFrameCopy, p0, maxFeatures, qualityLevel, minDistance, Mat(), 15, true, 0.04);

    // Create a mask image for drawing purposes
    Mat frameColor, mask;
    cvtColor(frame, frameColor, COLOR_GRAY2BGR);
    mask = Mat::zeros(frameColor.size(), frameColor.type());
    // Mat mask = Mat::zeros(prevFrameCopy.size(), prevFrameCopy.type());

    // calculate optical flow
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(prevFrameCopy, frameCopy, p0, p1, status, err, Size(15,15), 2, criteria);

    // Create some random colors
    vector<Scalar> colors;
    RNG rng(67);
    for(int i = 0; i < p0.size(); i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.emplace_back(r,g,b);
    }

    vector<Point2f> good_new;
    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i]) {
            line(frameColor, p1[i], p0[i], colors[i], 2);
            // circle(frameColor, p1[i], 5, colors[i], -1);
        }
    }

    result = frameColor.clone();
}