CC = nvcc

IFLAGS = -I/usr/include/opencv4
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc
FLAGS = -arch=sm_75 -std=c++17 -diag-suppress 611

FORMATTER = clang-format
FLOW_PROG = optflow
FLOW_OBJS = timing/stopwatch.o gpu/lk.o main.o
FLOW_SRCS = timing/stopwatch.cpp gpu/lk.cu main.cpp
FLOW_FORMAT = $(FLOW_SRCS) timing/stopwatch.hpp gpu/lk.cuh

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