#ifndef __SLIDEBUFFER_H__
#define __SLIDEBUFFER_H__

#include <cuda_runtime.h>
#include "cudaErrorChecking.h"

using namespace std;

/**
 * SlideBuffer class
 *
 * An implementation of a ringbuffer, allowing easy bi-directional movement
 * through a data set. It is designed for data arrays that are too large
 * to fit in memory, but which need to be "scrolled" through.
 */
class SlideBuffer
{

    protected:

        size_t  bufferBytes;  // The size of the buffer in bytes
        void   *bufferDevice; // The device buffer
        void   *bufferHost;   // The host buffer
        FILE   *srcStream;    // The file stream to be read
        size_t  offset;       // The starting offset into the buffer
        size_t  fileBytes;    // The size of the file in bytes

        /**
         * Utility function for reading bytes from a file stream,
         * but keeping the file position indicator at the *beginning*
         * of a block of data of size bufferBytes
         *
         * The appropriate parts of the data stream are read into bufferHost,
         * but which data are "appropriate" depends on whether the shift is
         * forwards (positive) or backwards (negative):
         *
         * Positive shift:
         * ```
         *   |---------------|
         *      |----------------|
         *      ^             ***
         *      |             Read-in data
         *      | New file position
         * ```
         *
         * Negative shift:
         * ```
         *      |----------------|
         *   |---------------|
         *   ^***
         *   |Read-in data
         *   | New file position
         * ```
         *
         * The value of slideAmount may be changed if the slide would overrun
         * the file boundaries.
         *
         * @param[in,out] slideAmount The amount to slide, in bytes
         */
        virtual void readStreamToHost( long *slideAmount );

    public:

        //
        // Getters
        //

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

        //
        // Setters
        //

        /**
         * Set the source file stream
         *
         * @param srcFile The name of the file to use as source
         */
        void setSrcFile( const char *srcFile, const char *mode = "r" );

        //
        // Manipulating data
        //

        /**
         * Fill the buffer and send to GPU.
         *
         * This reads in a whole buffer's worth of data from the file stream
         * and loads it to the GPU
         */
        void fillBuffer();

        /**
         * Copy the current contents of the GPU buffer to host
         */
        void pullBuffer();

        /**
         * Slide forwards or backwards by some number of bytes
         *
         * @param slideAmountBytes The number of bytes to slide. Positive
         *                         numbers mean slide forward, negative mean
         *                         slide back.
         */
        void slideAndRead( long slideAmountBytes );

        //
        // Constructor/Destructor
        //

        /**
         * Constructor for SlideBuffer class
         *
         * This constructor allocates memory on the CPU and the GPU.
         *
         * @param bytes   The size of the slide buffer in bytes
         * @param srcFile The name of the file from which the data will be
         *                read
         */
        SlideBuffer( size_t bytes, const char *srcFile = NULL, const char *mode = "r" );

        /**
         * Deconstructor for SlideBuffer class
         */
        ~SlideBuffer();
};

#endif
