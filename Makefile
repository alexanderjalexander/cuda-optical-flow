CC = nvcc

IFLAGS = -I/usr/include/opencv4
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc
FLAGS = -arch=sm_75 -std=c++17 -diag-suppress 611

FORMATTER = clang-format
FLOW_PROG = optflow
FLOW_CLS = main.cu
FLOW_FORMAT = ${FLOW_CLS}

all: ${FLOW_PROG}

${FLOW_PROG}: ${FLOW_CLS}
	${CC} ${FLAGS} ${IFLAGS} ${LDFLAGS} ${FLOW_CLS} -o ${FLOW_PROG}

clean:
	rm -f ${FLOW_PROG}

format:
	${FORMATTER} --style=file -i ${COMMON_CLS} ${FLOW_FORMAT}

format-15:
	${FORMATTER}-15 --style=file -i ${COMMON_CLS} ${FLOW_FORMAT}