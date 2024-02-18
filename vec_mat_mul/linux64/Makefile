TEST = test_exec
BENCH = bench_exec
CC = clang++-17
CFLAGS = -Wall -Wextra -O3 -mavx -mf16c

all: $(TEST) $(BENCH)

$(TEST): test.o lib.a
	$(CC) $^ $(CFLAGS) -fPIE -o $(TEST)

$(BENCH): bench.o lib.a
	$(CC) $^ $(CFLAGS) -fPIE -o $(BENCH) -L$(HOME)/dev/benchmark/build/src -lbenchmark -lpthread

test.o: test.cpp
	$(CC) -c $(CFLAGS) $< -o $@

bench.o: bench.cpp
	$(CC) -c $(CFLAGS) -I$(HOME)/dev/benchmark/include/ $< -o $@ 

lib.a: lib.o
	ar rcs $@ $<

lib.o: vec_mat_mul_linux64.asm
	nasm -f elf64 $< -o $@

clean:
	rm -f *.o *.a $(TEST) $(BENCH)

.PHONY: clean 
