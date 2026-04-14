#include "statistics.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

/**
 * @brief Private method to get the percentile of data. Hs[low] + (fandles even/odd sizes.
 *
 * @param sortedData Vector containing the already sorted data.
 * @param percentile The percentile to grab from the sorted data.
 * @return
 */
template <typename T>
T calculatePercentile(const std::vector<T>& sortedData, double percentile) {
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
    size_t low = static_cast<size_t>(std::floor(rank));
    size_t high = static_cast<size_t>(std::ceil(rank));
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
    std::stable_sort(exec.executionTimes.begin(), exec.executionTimes.end());

    size_t count = exec.executionTimes.size();
    std::chrono::duration<double> minTime = exec.executionTimes[0];
    std::chrono::duration<double> maxTime = exec.executionTimes[count - 1];
    double sum = std::reduce(
        exec.executionTimes.begin(),
        exec.executionTimes.end(),
        std::chrono::duration<double>(0.0)
    ).count();
    double mean = sum / count;

    double sqSum = std::reduce(
        exec.executionTimes.begin(),
        exec.executionTimes.end(),
        std::chrono::duration<double>(0.0),
        [mean](std::chrono::duration<double> a, std::chrono::duration<double> b) {
            double diff = b.count() - mean;
            return std::chrono::duration<double>(a.count() + diff * diff);
        }
    ).count();
    double stdDev = std::sqrt(sqSum / count);

    std::chrono::duration<double> perc25 = calculatePercentile(exec.executionTimes, 25.0);
    std::chrono::duration<double> median = calculatePercentile(exec.executionTimes, 50.0);
    std::chrono::duration<double> perc75 = calculatePercentile(exec.executionTimes, 75.0);

    std::cout << "==========" << functionName << " Execution Time Statistics" << "==========" << std::endl;
    std::printf("%-10s --> %10zu\n", "Count", count);
    std::printf("%-10s --> %10.4lf\n", "Minimum", minTime.count());
    std::printf("%-10s --> %10.4lf\n", "Maximum", maxTime.count());
    std::printf("%-10s --> %10.4lf\n", "Mean", mean);
    std::printf("%-10s --> %10.4lf\n", "Std. Dev.", stdDev);
    std::printf("%-10s --> %10.4lf\n", "25th Perc.", perc25.count());
    std::printf("%-10s --> %10.4lf\n", "50th Perc.", median.count());
    std::printf("%-10s --> %10.4lf\n", "75th Perc.", perc75.count());
}

/**
 * @brief
 *
 *
 *
 * @param onCPU
 * @param exec
 * @return
 */
int
recordStatsSparseLucasKanade(bool onCPU, ExecStats &exec)
{
    // TODO: COMPLETE
    return EXIT_SUCCESS;
}
