CC = nvcc

IFLAGS = -I/usr/include/opencv4
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc
FLAGS = -arch=sm_75 -std=c++17 -diag-suppress 611

FORMATTER = clang-format
FLOW_PROG = optflow
FLOW_OBJS = gpu/lk.o main.o
FLOW_SRCS = gpu/lk.cu main.cpp
FLOW_FORMAT = $(FLOW_SRCS) gpu/lk.cuh

all: ${FLOW_PROG}

${FLOW_PROG}: ${FLOW_OBJS}
	${CC} ${FLAGS} ${FLOW_OBJS} -o ${FLOW_PROG} ${LDFLAGS} 

main.o: main.cpp
	${CC} ${FLAGS} ${IFLAGS} -c $< -o $@

gpu/lk.o: gpu/lk.cu
	${CC} ${FLAGS} ${IFLAGS} -c $< -o $@

clean:
	rm -f ${FLOW_PROG} ${FLOW_OBJS}

format:
	${FORMATTER} --style=file -i ${COMMON_CLS} ${FLOW_FORMAT}

format-15:
	${FORMATTER}-15 --style=file -i ${COMMON_CLS} ${FLOW_FORMAT}