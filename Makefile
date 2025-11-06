CC=gcc
OPTS=-Ofast -march=native -flto
CFLAGS=$(OPTS) -std=c11 -g
LDFLAGS=-flto

CXX=g++
CXXFLAGS=$(OPTS) -std=c++17 -g

BINARIES=matrix matrix-cpp

.PHONY: all clean
all: $(BINARIES)

# C version
matrix: matrix.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# C++ version of matrix
matrix-cpp: matrix.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINARIES)
