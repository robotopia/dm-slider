#ifndef __SLIDEBUFFER_H__
#define __SLIDEBUFFER_H__

#include <cuda_runtime.h>

// Boilerplate code for checking CUDA functions
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    /* Wrapper function for GPU/CUDA error handling. Every CUDA call goes through
       this function. It will return a message giving your the error string,
       file name and line of the error. Aborts on error. */

    if (code != 0)
    {
        fprintf(stderr, "GPUAssert:: %s - %s (%d)\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__, true);}


/**
 * SlideBuffer class
 *
 * An implementation of a ringbuffer, allowing easy bi-directional movement
 * through a data set.
 */
class SlideBuffer {

    private:

        size_t  bufferBytes;  // The size of the buffer in bytes
        void   *bufferDevice; // The device buffer
        void   *bufferHost;   // The host buffer
        FILE   *srcStream;    // The file stream to be read
        size_t  modulo;       // The starting offset

    public:

        /**
         * Constructor for SlideBuffer class
         *
         * @param bytes The size of the slide buffer in bytes
         * @param fileStream The file stream from which the data will be read
         *
         * This constructor allocates memory on the CPU and the GPU.
         */
        SlideBuffer( size_t bytes, FILE *fileStream ) :
            bufferBytes{bytes},
            bufferDevice{NULL},
            bufferHost{NULL},
            srcStream{fileStream},
            modulo(0)
            {
                gpuErrchk( cudaMalloc( &bufferDevice, bytes ) );
                gpuErrchk( cudaMallocHost( &bufferHost, bytes ) );
            }

        /**
         * Deconstructor for SlideBuffer class
         */
        ~SlideBuffer() {
            gpuErrchk( cudaFree( bufferDevice ) );
            gpuErrchk( cudaFreeHost( bufferHost ) );
        }

        /**
         * Get the pointer to the device buffer
         *
         * @return Returns the slide buffer pointer
         */
        void *getDeviceBuffer() { return bufferDevice; }

        /**
         * Get the pointer to the device buffer
         *
         * @return Returns the slide buffer pointer
         */
        void *getHostBuffer() { return bufferHost; }

        /**
         * Get the buffer size (in bytes)
         *
         * @return Returns the size of the buffer (in bytes)
         */
        size_t getSize() { return bufferBytes; };

        /**
         * Slide forwards or backwards by some number of bytes
         *
         * @param slideAmountBytes The number of bytes to slide. Positive
         *                         numbers mean slide forward, negative mean
         *                         slide back.
         */
        void slide( int slideAmountBytes )
        {
            //
            // If slide amount is zero, do nothing
            if (slideAmountBytes == 0)
                return;

            // If slide amount is >= the size of the buffer, read in
            // a whole buffer amount
            if (slideAmountBytes >= bufferBytes)
            {
            }
        }
};

#endif
