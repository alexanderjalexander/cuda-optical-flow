#include "drawing.hpp"

#define MIN_CIRCLE_RAD 3
#define MIN_LINE_RAD 2

/**
 * Makes a size-specified, random array of colors.
 *
 * @param num The number of colors to generate.
 *
 * @returns A vector of Scalar values, representing a random color in RGB format.
 */
std::vector<cv::Scalar>
getRandomColors(int num)
{
    cv::RNG rng(67);

    std::vector<cv::Scalar> pt_colors;
    for (size_t i = 0; i < num; i++)
    {
        pt_colors.push_back(cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)));
    }

    return pt_colors;
}

/**
 * Draws the optical flow result onto the output frames.
 *
 * For compatibility reasons, use with OpenCV functions and CPU functions only.
 *
 * @param output The output frame.
 * @param mask The output mask to draw on.
 * @param p0 A vector representing the first frame's 2D point features.
 * @param p1 A vector representing the second frame's 2D point features.
 * @param status The status of the points.
 * @param colors An array of colors to draw the lines/arrows with.
 * @param drawContinuous Whether to draw a continuous line on the image mask, or to draw an arrow from one point to the
 * next.
 */
void
drawSparseOpticalFlow(cv::Mat &output, cv::Mat &mask, const std::vector<cv::Point2f> &p0,
                      const std::vector<cv::Point2f> &p1, const std::vector<uchar> &status,
                      const std::vector<cv::Scalar> &colors, bool drawContinuous)
{
    int width = output.cols;
    int height = output.rows;
    int minMeasure = std::min(width, height);
    int circleRadius = std::max(MIN_CIRCLE_RAD, (minMeasure / 360) * MIN_CIRCLE_RAD);
    int lineRadius = std::max(MIN_LINE_RAD, (minMeasure / 360) * MIN_LINE_RAD);

    for (uint j = 0; j < p0.size(); j++)
    {
        if (status[j] == 1)
        {
            if (drawContinuous)
            {
                line(mask, p1[j], p0[j], colors[j], lineRadius);
                circle(output, p1[j], circleRadius, colors[j], -1);
            }
            else
            {
                arrowedLine(output, p0[j], p1[j], colors[j], 5, cv::LineTypes::LINE_AA, 0, .3);
            }
        }
    }

    // If we're drawing continuous lines throughout the whole thing.
    if (drawContinuous)
    {
        bitwise_or(output, mask, output);
    }
}

/**
 * Draws the optical flow result onto the output frames.
 *
 * For compatibility reasons, use with OpenCV functions and CPU functions only.
 *
 * @param output The output frame.
 * @param p0 A vector representing the first frame's 2D point features.
 * @param p1 A vector representing the second frame's 2D point features.
 * @param status The status of the points.
 */
void
drawDenseOpticalFlow(cv::Mat &output, const std::vector<cv::Point2f> &p0, const std::vector<cv::Point2f> &p1,
                     const std::vector<uchar> &status)
{
    for (size_t i = 0; i < p0.size(); i++)
    {
        if (status[i] == 1)
        {
            float dx = p1[i].x - p0[i].x;
            float dy = p1[i].y - p0[i].y;

            float magnitude = std::sqrtf(dx * dx + dy * dy);
            float angle = std::atan2f(dy, dx);

            float hue = angle * (180.0f / CV_PI) / 2.0f;

            float val = std::min(magnitude * 10.0f, 255.0f);

            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, val));
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

            cv::circle(output, p1[i], 1, bgr.at<cv::Vec3b>(0, 0), -1);
        }
    }
}

/**
 * Draws the optical flow result onto the output frames.
 *
 * For compatibility reasons, use with GPU CUDA functions only.
 *
 * @param output The output frame.
 * @param mask The output mask to draw on.
 * @param prevFeatures A vector representing the first frame's 2D point features, with an additional status field.
 * @param features A vector representing the second frame's 2D point features, with an additional status field.
 * @param featureCount The number of features we have in total.
 * @param colors An array of colors to draw the lines/arrows with.
 * @param drawContinuous Whether to draw a continuous line on the image mask, or to draw an arrow from one point to the
 * next.
 */
void
drawSparseOpticalFlowGPU(cv::Mat &output, cv::Mat &mask, cv::Vec3f *prevFeatures, cv::Vec3f *features, int featureCount,
                         const std::vector<cv::Scalar> &colors, bool drawContinuous)
{
    int width = output.cols;
    int height = output.rows;
    int minMeasure = std::min(width, height);
    int circleRadius = std::max(MIN_CIRCLE_RAD, (minMeasure / 360) * MIN_CIRCLE_RAD);
    int lineRadius = std::max(MIN_LINE_RAD, (minMeasure / 360) * MIN_LINE_RAD);

    for (uint i = 0; i < featureCount; i++)
    {
        if (features[i][2] == 1)
        {
            cv::Point2f p0(prevFeatures[i][0], prevFeatures[i][1]);
            cv::Point2f p1(features[i][0], features[i][1]);

            if (drawContinuous)
            {
                line(mask, p1, p0, colors[i], lineRadius);
                circle(output, p1, circleRadius, colors[i], -1);
            }
            else
            {
                arrowedLine(output, p0, p1, colors[i], 5, cv::LineTypes::LINE_AA, 0, .3);
            }
        }
    }

    // If we're drawing continuous lines throughout the whole thing.
    if (drawContinuous)
    {
        bitwise_or(output, mask, output);
    }
}
