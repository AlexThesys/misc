TARGET = exec
CC = gcc
TYPE=RELEASE
CFLAGS = -Wall -Wextra
ifeq ($(TYPE),RELEASE)
	CFLAGS+= -O3 -DNDEBUG
else
	CFLAGS+= -O0 -g
endif

LDFLAGS = -lpthread -lz

$(TARGET) : main.c
	$(CC) $^ $(CFLAGS) -o $(TARGET) $(LDFLAGS)	

clean:
	rm -f *.o *.a $(TARGET)

.PHONY: clean 
