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

        size_t  bufferBytes;     // The size of the device buffer in bytes
        size_t  bufferHostBytes; // The size of the host buffer in bytes
        void   *bufferDevice;    // The device buffer
        void   *bufferHost;      // The host buffer
        FILE   *srcStream;       // The file stream to be read
        size_t  offset;          // The starting offset into the buffer
        size_t  fileBytes;       // The size of the file in bytes

        /**
         * Limit the given slide amount so that a slide of that
         * size would not exceed the file's boundaries.
         *
         * If the input slide amount is too big for the given file,
         * the output slide amount is set to the maximum possible size
         * without exceeding the file boundaries.
         *
         * @param slideAmount The input slide amount
         * @return The output slide amount
         */
        long limitSlideAmount( long slideAmount );

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

        /*
         * Get the current data position
         */
        virtual inline size_t getCurrentDataPos() { return ftell( srcStream ); }

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
         * @param mode    The I/O mode (same as fopen()
         */
        virtual void setSrcFile( const char *srcFile, const char *mode = "r" );

        //
        // Manipulating data
        //

        /**
         * Fill the buffer and send to GPU.
         *
         * This reads in a whole buffer's worth of data from the file stream
         * and loads it to the GPU.
         *
         * If `bytes == 0`, then read in as many bytes as possible (i.e. the
         * size of the host buffer).
         *
         * @param bytes The number of bytes to read from the file
         */
        void fillBuffer( size_t bytes = 0 );

        /**
         * Trim the non-data bytes from the host buffer
         *
         * The data bytes are packed without gaps at the beginning of the host buffer.
         * The order of the data bytes is retained.
         *
         * If `bytes == 0`, then use as many bytes as possible (i.e. the
         * size of the host buffer).
         *
         * @param bytes The number of bytes in the host buffer to include in the trim
         */
        virtual void trimBuffer( size_t bytes = 0 ) {} // Default behaviour is to do nothing = assumed already trimmed

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
         * @param mode    The I/O mode (same as fopen())
         */
        SlideBuffer( size_t bytes, const char *srcFile = NULL, const char *mode = "r", size_t hostBytes = 0 );

        /**
         * Deconstructor for SlideBuffer class
         */
        ~SlideBuffer();
};

#endif
