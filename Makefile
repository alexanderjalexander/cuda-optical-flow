CC = nvcc
FLAGS = -arch=sm_75 -std=c++17 -diag-suppress 611
IFLAGS = -I/usr/include/opencv4
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_video -lopencv_imgcodecs -lopencv_imgproc

FORMATTER = clang-format
FLOW_PROG = optflow
FLOW_OBJS = timing/stopwatch.o timing/statistics.o tracking/cpu.o tracking/gpu.o processing/video_io.o processing/drawing.o main.o
FLOW_SRCS = timing/stopwatch.cpp timing/statistics.cpp tracking/cpu.cpp tracking/gpu.cu processing/video_io.cpp processing/drawing.cpp main.cpp
FLOW_FORMAT = $(FLOW_SRCS) timing/stopwatch.hpp tracking/lucasKanade.hpp

all: ${FLOW_PROG}

${FLOW_PROG}: ${FLOW_OBJS}
	${CC} ${FLAGS} ${FLOW_OBJS} -o ${FLOW_PROG} ${LDFLAGS}

%.o: %.cpp
	${CC} ${FLAGS} ${IFLAGS} -c $< -o $@

%.o: %.cu
	${CC} ${FLAGS} ${IFLAGS} -c $< -o $@

clean:
	rm -f ${FLOW_PROG} ${FLOW_OBJS}

format:
	${FORMATTER} --style=file -i ${FLOW_FORMAT}

format-15:
	${FORMATTER}-15 --style=file -i ${FLOW_FORMAT}