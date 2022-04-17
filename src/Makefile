CC = nvcc

OBJECTS = SlideBuffer.o \
          VDIFBuffer.o

test: coherent_dedispersion
	prime-run ./$<

all: $(OBJECTS)

%.o: %.cpp
	nvcc -c -o $@ $<

coherent_dedispersion: coherent_dedispersion.cu vert.shader frag.shader
	nvcc -o $@ $< -lglfw -lGL -lGLU -lglut -lGLEW -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
