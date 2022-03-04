CC = nvcc

OBJECTS = SlideBuffer.o \
          VDIFBuffer.o

current: coherent_dedispersion

all: $(OBJECTS)

%.o: %.cpp
	nvcc -c -o $@ $<

coherent_dedispersion: coherent_dedispersion.cu
	nvcc -o $@ $< -lGL -lGLU -lglut
