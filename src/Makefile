CC = g++
NVCC = nvcc
NVCCLIBS = `pkg-config --libs gtk+-3.0 gl glu glut glew vdifio cufft`
LDLIBS = $(NVCCLIBS) \
		 `pkg-config --libs-only-l cuda cudart`
LDFLAGS = `pkg-config --libs-only-L cuda cudart`
NVCCFLAGS = -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
CFLAGS = -Wall -Wextra -O2 -march=native `pkg-config --cflags cuda cufft gtk+-3.0`


OBJECTS = dm-slider.o cohdd.o vdif.o ascii_header.o vds.o

run: dm-slider
	./$<

%.o: %.cu
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

dm-slider: $(OBJECTS)
	$(NVCC) -o $@ $^ $(NVCCLIBS)

#dm-slider.o: dm-slider.c vert.shader frag.shader
#	nvcc -o $@ $< -lglfw -lGL -lGLU -lglut -lGLEW -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored 
