CC = nvcc

OBJECTS = SlideBuffer.o \
          VDIFBuffer.o

all: $(OBJECTS)

%.o: %.cpp
	nvcc -c -o $@ $<
