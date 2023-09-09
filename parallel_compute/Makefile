TARGET = exec
TYPE=RELEASE
CC = gcc
CFLAGS = -Wall -Wextra 
ifeq ($(TYPE),RELEASE)
	CFLAGS+= -O3
else
	CFLAGS+= -O0 -g
endif
LDFLAGS = -lpthread

$(TARGET): main.c
	$(CC) $< $(CFLAGS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f *.o *.a $(TARGET)

.PHONY: clean 
