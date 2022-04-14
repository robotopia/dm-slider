NVCC = nvcc

TARGETS = SlideBuffer_tests \
		  VDIFBuffer_tests

all_tests: $(TARGETS)
	for prog in $(TARGETS); do ./$${prog}; done

SlideBuffer_tests: SlideBuffer_tests.o ../src/SlideBuffer.o
VDIFBuffer_tests: VDIFBuffer_tests.o ../src/VDIFBuffer.o ../src/SlideBuffer.o

vdif_benchmarks: vdif_benchmarks.o

../src/%.o:
	$(MAKE) -C ../src

%.o: %.cu
	$(NVCC) -c -o $@ $<
