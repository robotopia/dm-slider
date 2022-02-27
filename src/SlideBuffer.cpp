#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include "cudaErrorChecking.h"
#include "SlideBuffer.h"

using namespace std;

void SlideBuffer::readStreamToHost( long slideAmount )
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

void SlideBuffer::setSrcFile( const char *srcFile, const char *mode )
{
    srcStream = fopen( srcFile, mode );
    if (srcStream == NULL)
        fileBytes = 0;
    else
    {
        long curr_pos = ftell( srcStream );
        fseek( srcStream, 0, SEEK_END );
        fileBytes = ftell( srcStream );
        fseek( srcStream, curr_pos, SEEK_SET );
    }
}

void SlideBuffer::fillBuffer()
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

void SlideBuffer::pullBuffer()
{
    gpuErrchk( cudaMemcpy( bufferHost, bufferDevice, bufferBytes, cudaMemcpyDeviceToHost ) );
}

void SlideBuffer::slideAndRead( long slideAmountBytes )
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

SlideBuffer::SlideBuffer( size_t bytes, const char *srcFile, const char *mode ) :
    bufferBytes{bytes},
    bufferDevice{NULL},
    bufferHost{NULL},
    offset(0)
{
    setSrcFile( srcFile, mode );
    gpuErrchk( cudaMalloc( &bufferDevice, bytes ) );
    gpuErrchk( cudaMallocHost( &bufferHost, bytes ) );
}

SlideBuffer::~SlideBuffer() {
    gpuErrchk( cudaFree( bufferDevice ) );
    gpuErrchk( cudaFreeHost( bufferHost ) );

    if (srcStream != NULL)
        fclose( srcStream );
}
