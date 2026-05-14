#include "statistics.hpp"

#include <sys/mman.h>
#include <sys/wait.h>

#include "../tracking/lucasKanade.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;
using namespace chrono;

/**
 * @brief Private method to get the percentile of data. Hs[low] + (fandles even/odd sizes.
 *
 * @param sortedData Vector containing the already sorted data.
 * @param percentile The percentile to grab from the sorted data.
 * @return
 */
template <typename T> T
calculatePercentile(const vector<T> &sortedData, double percentile)
{
    if (sortedData.empty())
    {
        return static_cast<T>(0.0);
    }
    if (percentile <= 0.0)
    {
        return sortedData.front();
    }
    if (percentile >= 100.0)
    {
        return sortedData.back();
    }

    double rank = (percentile / 100.0) * (sortedData.size() - 1);
    size_t low = static_cast<size_t>(floor(rank));
    size_t high = static_cast<size_t>(ceil(rank));
    double fraction = rank - low;

    return sortedData[low] + (fraction * (sortedData[high] - sortedData[low]));
}

/**
 * @brief Calculates and prints out statistics of execution times.
 *
 * Takes in an ExecStats pointer, and calculates the count, mean, std. dev.,
 * minimum, 25th percentile, median, 75th percentile, and maximum.
 *
 * @param functionName the name of the function you're evaluating
 * @param exec a pointer to the ExecStats struct to evaluate statistics on.
 */
void
printStatistics(char *functionName, ExecStats &exec)
{
    stable_sort(exec.executionTimes.begin(), exec.executionTimes.end());

    size_t count = exec.executionTimes.size();
    duration<double> minTime = exec.executionTimes[0];
    duration<double> maxTime = exec.executionTimes[count - 1];
    double sum = reduce(exec.executionTimes.begin(), exec.executionTimes.end(), duration<double>(0.0)).count();
    double mean = sum / count;

    double sqSum = reduce(exec.executionTimes.begin(), exec.executionTimes.end(), duration<double>(0.0),
                          [mean](duration<double> a, duration<double> b)
                          {
                              double diff = b.count() - mean;
                              return duration<double>(a.count() + diff * diff);
                          })
                       .count();
    double stdDev = sqrt(sqSum / count);

    duration<double> perc25 = calculatePercentile(exec.executionTimes, 25.0);
    duration<double> median = calculatePercentile(exec.executionTimes, 50.0);
    duration<double> perc75 = calculatePercentile(exec.executionTimes, 75.0);

    cout << "========== " << functionName << " Execution Time Statistics" << " ==========" << endl;
    printf("%-10s --> %10zu\n", "Count", count);
    printf("%-10s --> %10.4lf\n", "Minimum", minTime.count());
    printf("%-10s --> %10.4lf\n", "Maximum", maxTime.count());
    printf("%-10s --> %10.4lf\n", "Mean", mean);
    printf("%-10s --> %10.4lf\n", "Std. Dev.", stdDev);
    printf("%-10s --> %10.4lf\n", "25th Perc.", perc25.count());
    printf("%-10s --> %10.4lf\n", "50th Perc.", median.count());
    printf("%-10s --> %10.4lf\n", "75th Perc.", perc75.count());
}

/**
 * @brief Records statistics for CPU and GPU LK on a video.
 *
 * Performs a long-ran statistics test to obtain relevant metrics to each LK version for this program.
 * Obtains mean/min/max/count/std.dev/etc., and displays them.
 *
 * @param onCPU Whether to run the test on the CPU LK algorithm or GPU LK algorithm.
 * @param progFlags Flags passed in at program start.
 * @param video The video to run the test on.
 * @return EXIT_FAILURE or EXIT_SUCCESS
 */
int
recordStatsSparseLucasKanade(bool onCPU, ProgramFlags progFlags, VideoInfo &video)
{
    vector<duration<double>> execTimesVec;
    execTimesVec.reserve(STATISTICS_ITERATIONS);

    char functionNameSparseLKCPU[] = "sparseLucasKanadeCPU";
    char functionNameSparseLKGPU[] = "sparseLucasKanadeGPU";

    std::cout << "===== "
              << "Starting " << STATISTICS_ITERATIONS << " iterations of "
              << (onCPU ? functionNameSparseLKCPU : functionNameSparseLKGPU) << " =====" << std::endl;

    for (int i = 0; i < STATISTICS_ITERATIONS; i++)
    {
        video.frames.resetCapture();
        auto startTime = high_resolution_clock::now();
        if (onCPU)
        {
            sparseLucasKanadeCPU(video, progFlags.mipMap);
        }
        else
        {
            if (progFlags.textureMem)
            {
                // sparseLucasKanadeGPUTex(video);
            }
            else if (progFlags.mipMap)
            {
                // sparseLucasKanadeGPUMip(video);
            }
            else
            {
                sparseLucasKanadeGPU(video);
            }
        }
        auto stopTime = high_resolution_clock::now();
        duration<double> sec = stopTime - startTime;

        std::fprintf(stdout, "--> %*d. ", (int)floor(log10(STATISTICS_ITERATIONS) + 1), i + 1);
        std::cout << sec.count() << " seconds" << std::endl;
        execTimesVec.push_back(sec);

        video.outputFrames.clear();
    }

    ExecStats execStats = {execTimesVec};

    if (onCPU)
    {
        printStatistics(functionNameSparseLKCPU, execStats);
    }
    else
    {
        printStatistics(functionNameSparseLKGPU, execStats);
    }

    return EXIT_SUCCESS;
}
