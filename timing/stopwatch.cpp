#include "stopwatch.hpp"

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using std::chrono::duration;

time_point<high_resolution_clock> startTime;
time_point<high_resolution_clock> stopTime;

void
startStopwatch()
{
    startTime = high_resolution_clock::now();
}

double
stopStopwatch()
{
    stopTime = high_resolution_clock::now();
    duration<double> sec = stopTime - startTime;
    std::cout << "Completed in " << sec.count() << " seconds." << std::endl;

    return sec.count();
}