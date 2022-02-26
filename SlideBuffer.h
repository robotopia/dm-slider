#ifndef __SLIDEBUFFER_H__
#define __SLIDEBUFFER_H__

#include <cuda_runtime.h>

typedef enum onDeviceOrHost_t {
    ON_DEVICE,
    ON_HOST
} onDeviceOrHost;

class SlideBuffer {

    private:

        size_t  bufferBytes; // The size of the buffer in bytes
        bool    isDevice;    // true = buffer on GPU, false = buffer on CPU
        void   *buffer;      // the buffer itself

    public:

        /**
         * Constructor for SlideBuffer class
         *
         * @param bytes The size of the slide buffer in bytes
         * @param isDevice Set true for GPU memory, false for CPU memory
         */
        SlideBuffer( size_t bytes, onDeviceOrHost location = ON_DEVICE ) :
            bufferBytes{bytes},
            buffer{NULL},
            isDevice{true}
            {
                isDevice = (location == ON_DEVICE);

                if (isDevice)
                    cudaMalloc( &buffer, bytes );
                else
                    cudaMallocHost( &buffer, bytes );
            }

        /**
         * Deconstructor for SlideBuffer class
         */
        ~SlideBuffer() {
            if (isDevice)
                cudaFree( buffer );
            else
                cudaFreeHost( buffer );
        }

        /**
         * Get the pointer to the buffer
         *
         * @return Returns the slide buffer pointer
         */
        void *getBuffer() { return buffer; }

        /**
         * Get the buffer size (in bytes)
         *
         * @return Returns the size of the buffer (in bytes)
         */
        size_t getSize() { return bufferBytes; };
};

#endif
