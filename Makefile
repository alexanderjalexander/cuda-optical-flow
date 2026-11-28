# Compilers
CC = nvcc

# OpenCV Version Selection & Version Pinning
OPENCV_VERSION = 5.1.0
OPENCV_PREFIX  = /opt/opencv-$(OPENCV_VERSION)
OPENCV_INC = $(OPENCV_PREFIX)/include/opencv$(word 1,$(subst ., ,$(OPENCV_VERSION)))
OPENCV_LIB = $(OPENCV_PREFIX)/lib64
OPENCV_MODULES = core highgui videoio video imgcodecs imgproc features

# Compilation Configuration
FLAGS = -arch=sm_75 -std=c++17 -diag-suppress 611 -MMD -MP
IFLAGS = -I$(OPENCV_INC)
IFLAGS = -I$(OPENCV_INC)
LDFLAGS = -L$(OPENCV_LIB) -Xlinker -rpath -Xlinker $(OPENCV_LIB)
LDLIBS = $(addprefix -lopencv_,$(OPENCV_MODULES))

# Formatters & Build Defaults
FORMATTER = clang-format
BUILD_DIR = build
FLOW_PROG = optflow

# Processing
PROCESSING_SRCS = processing/video_io.cpp processing/drawing.cpp processing/frame_buffer.cpp
PROCESSING_OBJS = $(addprefix $(BUILD_DIR)/, $(patsubst %.cu,%.o, $(patsubst %.cpp,%.o, $(PROCESSING_SRCS))))
PROCESSING_HDRS = processing/video_io.hpp processing/drawing.hpp processing/frame_buffer.hpp

# Timing
TIMING_SRCS = timing/stopwatch.cpp timing/statistics.cpp
TIMING_OBJS = $(addprefix $(BUILD_DIR)/, $(patsubst %.cu,%.o, $(patsubst %.cpp,%.o, $(TIMING_SRCS))))
TIMING_HDRS = timing/stopwatch.hpp timing/statistics.hpp

# Tracking
TRACKING_SRCS = tracking/correctness.cpp tracking/cpu.cpp tracking/gpu.cu #tracking/gpu_tex.cu tracking/gpu_mip.cu
TRACKING_OBJS = $(addprefix $(BUILD_DIR)/, $(patsubst %.cu,%.o, $(patsubst %.cpp,%.o, $(TRACKING_SRCS))))
TRACKING_HDRS = tracking/lucasKanade.hpp tracking/gpu_utilities.cuh

# Final Compiled Obj's/Src's
FLOW_OBJS = $(PROCESSING_OBJS) $(TIMING_OBJS) $(TRACKING_OBJS) $(BUILD_DIR)/main.o
FLOW_SRCS = $(PROCESSING_SRCS) $(TIMING_SRCS) $(TRACKING_SRCS) main.cpp
FLOW_HDRS = $(PROCESSING_HDRS) $(TIMING_HDRS) $(TRACKING_HDRS)

FLOW_FORMAT = $(FLOW_SRCS) $(FLOW_HDRS)

# Make all the directories as we need to
$(shell mkdir -p $(sort $(dir $(FLOW_OBJS))))
$(shell touch $(FLOW_OBJS:.o=.d))

all: $(FLOW_PROG)

$(FLOW_PROG): $(FLOW_OBJS)
	$(CC) $(FLAGS) $(FLOW_OBJS) -o $(FLOW_PROG) $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/%.o: %.cpp
	$(CC) $(FLAGS) $(IFLAGS) -MF $(@:.o=.d) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	$(CC) $(FLAGS) $(IFLAGS) -MF $(@:.o=.d) -c $< -o $@

clean:
	rm -f $(FLOW_PROG)
	rm -rf $(BUILD_DIR)

format:
	$(FORMATTER) --style=file --Wno-error=unknown -i $(FLOW_FORMAT)

format-15:
	$(FORMATTER)-15 --style=file --Wno-error=unknown -i $(FLOW_FORMAT)

-include $(FLOW_OBJS:.o=.d)
