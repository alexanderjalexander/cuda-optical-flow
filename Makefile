CC = nvcc
FLAGS = -arch=sm_75

FORMATTER = clang-format
FLOW_PROG = optflow
FLOW_CLS = main.cu
FLOW_FORMAT = ${FLOW_CLS}

all: ${FLOW_PROG}

${FLOW_PROG}: ${FLOW_CLS}
	${CC} ${FLAGS} ${FLOW_CLS} -o ${FLOW_PROG}

clean:
	rm ${FLOW_PROG}

format:
	${FORMATTER} --style=file -i ${COMMON_CLS} ${FLOW_FORMAT}

format-15:
	${FORMATTER}-15 --style=file -i ${COMMON_CLS} ${FLOW_FORMAT}