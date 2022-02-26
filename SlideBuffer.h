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
        size_t  offset;       // The starting offset into the buffer
        size_t  fileBytes;    // The size of the file in bytes

        /**
         * Utility function for reading bytes from a file stream,
         * but keeping the file position indicator at the *beginning*
         * of a block of data of size bufferBytes
         *
         * @param slideAmount The amount to slide, in bytes
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
         * This function does not check whether or not beginning-of-file, or
         * EOF is reached.
         */
        void readStreamToHost( long slideAmount )
        {
            if (slideAmount == 0)
                return;

            size_t readAmount = abs(slideAmount);
            if (readAmount > bufferBytes)
                readAmount = bufferBytes;

            size_t curr_pos  = ftell( srcStream );
            size_t read_pos  = curr_pos + (slideAmount > 0 && slideAmount <= bufferBytes ? bufferBytes : slideAmount);
            size_t final_pos = curr_pos + slideAmount;

#ifdef DEBUG
            fprintf( stderr, "slideAmount = %ld\n", slideAmount );
            fprintf( stderr, "curr_pos = %ld\n", curr_pos );
            fprintf( stderr, "read_pos = %ld\n", read_pos );
            fprintf( stderr, "final_pos = %ld\n", final_pos );
            fprintf( stderr, "\n" );
#endif

            fseek( srcStream,  read_pos, SEEK_SET );       // Jump to the appropriate place
            fread( bufferHost, readAmount, 1, srcStream ); // Read in the data
            fseek( srcStream,  final_pos, SEEK_SET );      // Re-place the file position indicator
        }

    public:

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
         * Set the source file stream
         *
         * @param fileStream The file stream to use as source
         */
        void setSrcStream( FILE *fileStream ) {
            srcStream = fileStream;
            if (fileStream == NULL)
                fileBytes = 0;
            else
            {
                long curr_pos = ftell( srcStream );
                fseek( srcStream, 0, SEEK_END );
                fileBytes = ftell( srcStream );
                fseek( srcStream, curr_pos, SEEK_SET );
            }
        }

        /**
         * Fill the buffer and send to GPU.
         *
         * This reads in a whole buffer's worth of data from the file stream
         * and loads it to the GPU
         */
        void fillBuffer()
        {
            if (srcStream == NULL)
                return;

            // Read in the data
            fread( bufferHost, bufferBytes, 1, srcStream );
            fseek( srcStream, -bufferBytes, SEEK_CUR );

            // Send to GPU
            gpuErrchk( cudaMemcpy( bufferDevice, bufferHost, bufferBytes, cudaMemcpyHostToDevice ) );

            // Reset offset
            offset = 0;
        }

        /**
         * Copy the current contents of the GPU buffer to host
         */
        void pullBuffer()
        {
            gpuErrchk( cudaMemcpy( bufferHost, bufferDevice, bufferBytes, cudaMemcpyDeviceToHost ) );
        }

        /**
         * Slide forwards or backwards by some number of bytes
         *
         * @param slideAmountBytes The number of bytes to slide. Positive
         *                         numbers mean slide forward, negative mean
         *                         slide back.
         */
        void slideAndRead( long slideAmountBytes )
        {
            // Check that the source file stream is set
            if (srcStream == NULL)
                return;

            // Limit the slide amount so that it doesn't exceed the file
            // stream boundaries
            long curr_pos = ftell( srcStream );
            long lastAllowedPos = fileBytes - bufferBytes;

            if (curr_pos + slideAmountBytes < 0)
                slideAmountBytes = -curr_pos;
            else if (curr_pos + slideAmountBytes > lastAllowedPos)
                slideAmountBytes = lastAllowedPos - curr_pos;

            // If slide amount is zero, do nothing
            if (slideAmountBytes == 0)
                return;

            // Read in the new data (to bufferHost)
            readStreamToHost( slideAmountBytes );
            // (Now, bufferHost contains *only* the new data)

            // If absolute value of slide amount is >= the size of the buffer,
            // just read in a whole buffer amount
            if (abs(slideAmountBytes) >= bufferBytes)
            {
                // Copy the data to GPU
                gpuErrchk( cudaMemcpy( bufferDevice, bufferHost, bufferBytes, cudaMemcpyHostToDevice ) );

                // Reset the offset to zero, for one contiguous read
                offset = 0;

                // We're done, so return
                return;
            }

            // If we got this far, we have to implement the sliding action

            // Calculate where the final offset will be after we're done
            size_t newOffset = (offset + slideAmountBytes + bufferBytes) % bufferBytes;

            // Calculate the offset where we need to start writing (and get a
            // corresponding device pointer)
            size_t writeOffset = (slideAmountBytes < 0 ? newOffset : offset);
            char *d_ptr = (char *)bufferDevice + writeOffset;

            // Calculate how many bytes will be written
            size_t writeAmount = slideAmountBytes * (slideAmountBytes < 0 ? -1 : 1);

            // Can we get away with a single write?
            if (writeOffset + writeAmount < bufferBytes)
            {
                gpuErrchk( cudaMemcpy( d_ptr, bufferHost, writeAmount, cudaMemcpyHostToDevice ) );
            }
            else
            {
                // Do two writes then, one to the end of the buffer, and one to the front
                size_t writeEndAmount   = bufferBytes - writeOffset;
                size_t writeFrontAmount = writeAmount - writeEndAmount;

                char *ptrEnd   = (char *)bufferHost;
                char *ptrFront = ptrEnd + writeEndAmount;

                gpuErrchk( cudaMemcpy( d_ptr, ptrEnd, writeEndAmount, cudaMemcpyHostToDevice ) );
                gpuErrchk( cudaMemcpy( bufferDevice, ptrFront, writeFrontAmount, cudaMemcpyHostToDevice ) );
            }

            // Update the offset
            offset = newOffset;
        }

        /**
         * Constructor for SlideBuffer class
         *
         * @param bytes The size of the slide buffer in bytes
         * @param fileStream The file stream from which the data will be read
         *
         * This constructor allocates memory on the CPU and the GPU.
         */
        SlideBuffer( size_t bytes, FILE *fileStream = NULL ) :
            bufferBytes{bytes},
            bufferDevice{NULL},
            bufferHost{NULL},
            offset(0)
            {
                setSrcStream( fileStream );
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

};

#endif
