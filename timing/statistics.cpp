#include "statistics.hpp"
#include "../tracking/lucasKanade.hpp"

#include <sys/mman.h>
#include <sys/wait.h>

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
template <typename T>
T calculatePercentile(const vector<T>& sortedData, double percentile) {
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
    double sum = reduce(
        exec.executionTimes.begin(),
        exec.executionTimes.end(),
        duration<double>(0.0)
    ).count();
    double mean = sum / count;

    double sqSum = reduce(
        exec.executionTimes.begin(),
        exec.executionTimes.end(),
        duration<double>(0.0),
        [mean](duration<double> a, duration<double> b) {
            double diff = b.count() - mean;
            return duration<double>(a.count() + diff * diff);
        }
    ).count();
    double stdDev = sqrt(sqSum / count);

    duration<double> perc25 = calculatePercentile(exec.executionTimes, 25.0);
    duration<double> median = calculatePercentile(exec.executionTimes, 50.0);
    duration<double> perc75 = calculatePercentile(exec.executionTimes, 75.0);

    cout << "==========" << functionName << " Execution Time Statistics" << "==========" << endl;
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
 * @brief
 *
 *
 *
 * @param onCPU
 * @param exec
 * @return
 */
int
recordStatsSparseLucasKanade(bool onCPU, VideoInfo &video, ExecStats &exec)
{
    vector<pid_t> children = vector<pid_t>(STATISTICS_ITERATIONS);
    unsigned long execTimesSize = sizeof(duration<double>) * STATISTICS_ITERATIONS;
    duration<double> *execTimes = (duration<double>*)mmap(
        NULL,
        execTimesSize,
        PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS,
        -1,
        0
    );


    for (int i = 0; i < STATISTICS_ITERATIONS; i++)
    {
        pid_t pid = fork();
        if (pid < 0)
        {
            std::cerr << "Fork failed: " << strerror(errno) << std::endl;
            return EXIT_FAILURE;
        }
        else if (pid == 0)
        {
            auto startTime = high_resolution_clock::now();
            if (onCPU)
            {
                sparseLucasKanadeCPU(video);
            }
            else
            {
                sparseLucasKanadeGPU(video);
            }
            auto stopTime = high_resolution_clock::now();
            duration<double> sec = stopTime - startTime;
            execTimes[i] = sec;
            exit(EXIT_SUCCESS);
        }
        else
        {
            children.push_back(pid);
        }
    }

    for (int i = 0; i < children.size(); i++)
    {
        int status;
        if (waitpid(children[i], &status, 0) < 0)
        {
            std::cerr << "Child at PID " << children[i] << " failed with status " << WEXITSTATUS(status) << std::endl;
        }
    }

    vector<duration<double>> execTimesVec(
        static_cast<duration<double>*>(execTimes),
        static_cast<duration<double>*>(execTimes) + execTimesSize
    );

    ExecStats execStats = {
        execTimesVec
    };

    munmap(execTimes, execTimesSize);

    char functionNameSparseLKCPU[] = "sparseLucasKanadeCPU";
    char functionNameSparseLKGPU[] = "sparseLucasKanadeGPU";

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
