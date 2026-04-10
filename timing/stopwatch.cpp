#include "stopwatch.hpp"

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using std::chrono::duration;

time_point<high_resolution_clock> startTime;
time_point<high_resolution_clock> stopTime;

/**
 * Starts a stopwatch, marking the current time.
 */
void
startStopwatch()
{
    startTime = high_resolution_clock::now();
}

/**
 * Stops a stopwatch if started.
 *
 * Returns the difference between stopTime and startTime, printing the number
 * of seconds the operation took.
 *
 * @returns a double of the number of seconds.
 */
double
stopStopwatch()
{
    stopTime = high_resolution_clock::now();
    duration<double> sec = stopTime - startTime;
    std::cout << "Completed in " << sec.count() << " seconds." << std::endl;

    return sec.count();
}
