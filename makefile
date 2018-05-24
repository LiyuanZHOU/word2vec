CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -Wall -funroll-loops -Wno-unused-result

all: incre_train incre_net lib

lib : compute-accuracy.h

incre_train : incre_train.c
	$(CC) incre_train.c -o incre_train $(CFLAGS)
	chmod +x *.sh

incre_net : incre_net.c
	$(CC) incre_net.c -o incre_net $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf incre_train incre_train
