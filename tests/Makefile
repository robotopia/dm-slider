CC = nvcc

tests: SlideBuffer_tests
	./$<

SlideBuffer_tests: SlideBuffer_tests.o ../src/SlideBuffer.o

../src/%.o:
	$(MAKE) -C ../src

%.o: %.cu
	$(CC) -c -o $@ $<
