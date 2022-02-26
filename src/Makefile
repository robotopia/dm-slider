CC = nvcc

OBJECTS = SlideBuffer.o

all: $(OBJECTS)

%.o: %.cpp
	nvcc -c -o $@ $<
