#include "drawing.hpp"

vector<Scalar> getRandomColors(int num)
{
    RNG rng;

    vector<Scalar> pt_colors;
    for (size_t i = 0; i < num; i++) {
        pt_colors.push_back(Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)));
    }

    return pt_colors;
}

void
drawOpticalFlow(
    Mat &output,
    Mat &mask,
    const vector<Point2f> &p0,
    const vector<Point2f> &p1,
    const vector<uchar> &status,
    const vector<Scalar> &colors,
    bool drawContinuous)
{
    for (uint j = 0; j < p0.size(); j++)
    {
        if (status[j] == 1)
        {
            if (drawContinuous) {
                line(mask, p1[j], p0[j], colors[j], 2);
                circle(output, p1[j], 5, colors[j], -1);
            } else {
                arrowedLine(output, p0[j], p1[j], colors[j], 5, cv::LineTypes::LINE_AA, 0, .3);
            }
        }
    }

    // If we're drawing continuous lines throughout the whole thing.
    if (drawContinuous) {
        bitwise_or(output, mask, output);
    }
}

void
drawOpticalFlowGPU(
    Mat &output,
    Mat &mask,
    Vec3f *prevFeatures,
    Vec3f *features,
    int featureCount,
    const vector<Scalar>& colors,
    bool drawContinuous)
{
    for (uint i = 0; i < featureCount; i++)
    {
        if (features[i][2] == 1)
        {
            Point2f p0(features[i][0], features[i][1]);
            Point2f p1(features[i][0], features[i][1]);

            if (drawContinuous) {
                line(mask, p1, p0, colors[i], 2);
                circle(output, p1, 5, colors[i], -1);
            } else {
                arrowedLine(output, p0, p1, colors[i], 5, cv::LineTypes::LINE_AA, 0, .3);
            }
        }
    }

    // If we're drawing continuous lines throughout the whole thing.
    if (drawContinuous) {
        bitwise_or(output, mask, output);
    }
}