CC = nvcc

IFLAGS = -I/usr/include/opencv4
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc
FLAGS = -arch=sm_75 -std=c++17 -diag-suppress 611

FORMATTER = clang-format
FLOW_PROG = optflow
FLOW_SRCS = gpu/kernels.cu main.cpp
FLOW_FORMAT = $(FLOW_SRCS)

all: ${FLOW_PROG}

${FLOW_PROG}: ${FLOW_SRCS}
	${CC} ${FLAGS} ${IFLAGS} ${LDFLAGS} ${FLOW_SRCS} -o ${FLOW_PROG}

clean:
	rm -f ${FLOW_PROG}

format:
	${FORMATTER} --style=file -i ${COMMON_CLS} ${FLOW_FORMAT}

format-15:
	${FORMATTER}-15 --style=file -i ${COMMON_CLS} ${FLOW_FORMAT}