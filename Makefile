CC = nvcc
FLAGS = -arch=sm_75 -std=c++17 -diag-suppress 611 -MMD -MP
IFLAGS = -I/usr/include/opencv4
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_video -lopencv_imgcodecs -lopencv_imgproc

FORMATTER = clang-format
FLOW_PROG = optflow

PROCESSING_OBJS = processing/video_io.o processing/drawing.o
PROCESSING_SRCS = processing/video_io.cpp processing/drawing.cpp
PROCESSING_HDRS = processing/video_io.hpp processing/drawing.hpp

TIMING_OBJS = timing/stopwatch.o timing/statistics.o
TIMING_SRCS = timing/stopwatch.cpp timing/statistics.cpp
TIMING_HDRS = timing/stopwatch.hpp timing/statistics.hpp

TRACKING_OBJS = tracking/cpu.o tracking/gpu.o
TRACKING_SRCS = tracking/cpu.cpp tracking/gpu.cu
TRACKING_HDRS = tracking/lucasKanade.hpp tracking/gpu_utilities.cuh

FLOW_OBJS = $(PROCESSING_OBJS) $(TIMING_OBJS) $(TRACKING_OBJS) main.o
FLOW_SRCS = $(PROCESSING_SRCS) $(TIMING_SRCS) $(TRACKING_SRCS) main.cpp
FLOW_HDRS = $(PROCESSING_HDRS) $(TIMING_HDRS) $(TRACKING_HDRS)

FLOW_FORMAT = $(FLOW_SRCS) $(FLOW_HDRS)

all: ${FLOW_PROG}

${FLOW_PROG}: ${FLOW_OBJS}
	${CC} ${FLAGS} ${FLOW_OBJS} -o ${FLOW_PROG} ${LDFLAGS}

%.o: %.cpp
	${CC} ${FLAGS} ${IFLAGS} -MF $(@:.o=.d) -c $< -o $@

%.o: %.cu
	${CC} ${FLAGS} ${IFLAGS} -MF $(@:.o=.d) -c $< -o $@

clean:
	rm -f ${FLOW_PROG} ${FLOW_OBJS} $(FLOW_OBJS:.o=.d) main.d

format:
	${FORMATTER} --style=file -i ${FLOW_FORMAT}

format-15:
	${FORMATTER}-15 --style=file -i ${FLOW_FORMAT}


# Include automatically generated dependency files
-include $(FLOW_OBJS:.o=.d) main.d
