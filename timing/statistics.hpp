#ifndef STATISTICS_H
#define STATISTICS_H

#include "../processing/video_io.hpp"
#include "../flags.hpp"

#include <chrono>
#include <vector>

/**
 * How many iterations the recordStats function(s) should record execution
 * times for.
 */
#define STATISTICS_ITERATIONS 100

struct ExecStats
{
    std::vector<std::chrono::duration<double>> executionTimes;
};

void printStatistics(char *functionName, ExecStats &exec);
int recordStatsSparseLucasKanade(bool onCPU, ProgramFlags progFlags, VideoInfo &video);

#endif
