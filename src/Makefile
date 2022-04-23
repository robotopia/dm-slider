CC = g++
NVCC = nvcc
NVCCLIBS = -lGL -lGLU -lglut -lGLEW `pkg-config --libs gtk+-3.0`
LDLIBS = $(NVCCLIBS) -lcuda -lcudart
LDFLAGS = `pkg-config --libs-only-L cuda cudart`
NVCCFLAGS = -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
CFLAGS = -Wall -Wextra -O2 -march=native `pkg-config --cflags cuda` `pkg-config --cflags gtk+-3.0`


OBJECTS = dm-slider.o cohdd.o

test: dm-slider
	./$<

all: $(OBJECTS)

%.o: %.cu
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

dm-slider: dm-slider.o cohdd.o
	$(NVCC) -o $@ $^ $(NVCCLIBS)

#dm-slider.o: dm-slider.c vert.shader frag.shader
#	nvcc -o $@ $< -lglfw -lGL -lGLU -lglut -lGLEW -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored 
