CC = nvcc
FLAGS = -arch=sm_75 -std=c++17 -diag-suppress 611 -MMD -MP
IFLAGS = -I/usr/include/opencv4
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_video -lopencv_imgcodecs -lopencv_imgproc

FORMATTER = clang-format
BUILD_DIR = build
FLOW_PROG = optflow

PROCESSING_OBJS = $(BUILD_DIR)/processing/video_io.o $(BUILD_DIR)/processing/drawing.o
PROCESSING_SRCS = processing/video_io.cpp processing/drawing.cpp
PROCESSING_HDRS = processing/video_io.hpp processing/drawing.hpp

TIMING_OBJS = $(BUILD_DIR)/timing/stopwatch.o $(BUILD_DIR)/timing/statistics.o
TIMING_SRCS = timing/stopwatch.cpp timing/statistics.cpp
TIMING_HDRS = timing/stopwatch.hpp timing/statistics.hpp

TRACKING_OBJS = $(BUILD_DIR)/tracking/cpu.o $(BUILD_DIR)/tracking/gpu.o
TRACKING_SRCS = tracking/cpu.cpp tracking/gpu.cu
TRACKING_HDRS = tracking/lucasKanade.hpp tracking/gpu_utilities.cuh

FLOW_OBJS = $(PROCESSING_OBJS) $(TIMING_OBJS) $(TRACKING_OBJS) $(BUILD_DIR)/main.o
FLOW_SRCS = $(PROCESSING_SRCS) $(TIMING_SRCS) $(TRACKING_SRCS) main.cpp
FLOW_HDRS = $(PROCESSING_HDRS) $(TIMING_HDRS) $(TRACKING_HDRS)

FLOW_FORMAT = $(FLOW_SRCS) $(FLOW_HDRS)

# Make all the directories as we need to
$(shell mkdir -p $(sort $(dir $(FLOW_OBJS))))
$(shell touch $(FLOW_OBJS:.o=.d))

all: ${FLOW_PROG}

${FLOW_PROG}: ${FLOW_OBJS}
	${CC} ${FLAGS} ${FLOW_OBJS} -o ${FLOW_PROG} ${LDFLAGS}

$(BUILD_DIR)/%.o: %.cpp
	${CC} ${FLAGS} ${IFLAGS} -MF $(@:.o=.d) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	${CC} ${FLAGS} ${IFLAGS} -MF $(@:.o=.d) -c $< -o $@

clean:
	rm -f ${FLOW_PROG}
	rm -rf ${BUILD_DIR}

format:
	${FORMATTER} --style=file -i ${FLOW_FORMAT}

format-15:
	${FORMATTER}-15 --style=file -i ${FLOW_FORMAT}

-include $(FLOW_OBJS:.o=.d)