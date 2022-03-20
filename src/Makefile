CC = nvcc

OBJECTS = SlideBuffer.o \
          VDIFBuffer.o

current: coherent_dedispersion

all: $(OBJECTS)

%.o: %.cpp
	nvcc -c -o $@ $<

coherent_dedispersion: coherent_dedispersion.cu
	nvcc -o $@ $< -lglfw -lGL -lGLU -lglut -lGLEW -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
