#include <cstdlib>
#include <iostream>
#include <vdifio.h>
#include "VDIFBuffer.h"

using namespace std;

long VDIFBuffer::dataPosToFilePos( long pos )
{
    // Only meaningful if a VDIF file has been loaded
    if (srcStream == NULL)
        return 0L;

    long frame = pos / VDIFDataBytes;

    return pos + frame*VDIFHeaderBytes;
}

long VDIFBuffer::filePosToDataPos( long pos )
{
    // Only meaningful if a VDIF file has been loaded
    if (srcStream == NULL)
        return 0L;

    // If a negative pos is given, use the current position of
    // srcStream as a default.
    if (pos < 0)
        pos = ftell( srcStream );

    long frame = pos / VDIFFrameBytes;

    // If the file pos is somewhere in a frame header,
    // pretend that it's pointing to the first data byte
    // in that frame
    long frame_pos      = pos % VDIFFrameBytes;
    long frame_data_pos = (frame_pos <= VDIFHeaderBytes ?
            0 : frame_pos - VDIFHeaderBytes);

    return frame*VDIFDataBytes + frame_data_pos;
}

void VDIFBuffer::readStreamToHost( long *slideAmount )
{
    // Limit the slide amount so that it doesn't exceed the file
    // stream boundaries
    *slideAmount = limitSlideAmount( *slideAmount );

    // If slide amount is zero, do nothing
    if (*slideAmount == 0)
        return;

    size_t readAmount = abs(*slideAmount);
    if (readAmount > bufferBytes)
        readAmount = bufferBytes;

    size_t data_pos  = getCurrentDataPos();
    size_t read_pos  = data_pos + (*slideAmount > 0 && *slideAmount <= bufferBytes ? bufferBytes : *slideAmount);
    size_t final_pos = data_pos + *slideAmount;

#ifdef DEBUG
    fprintf( stderr, "slideAmount = %ld\n", *slideAmount );
    fprintf( stderr, "data_pos = %ld\n", data_pos );
    fprintf( stderr, "read_pos = %ld\n", read_pos );
    fprintf( stderr, "final_pos = %ld\n", final_pos );
    fprintf( stderr, "\n" );
#endif

    // Jump to the appropriate place
    fseek( srcStream,  dataPosToFilePos( read_pos ), SEEK_SET );

    // Read in the data (one frame at a time, necessarily)
    char *dest = (char *)bufferHost; // Start writing at the start
    vdif_header vhdr_dummy; // For consuming (and discarding) headers

    // The first frame to read might not be a whole frame, depending
    // on where the read_pos current is
    long frameReadAmount = VDIFDataBytes - read_pos % VDIFDataBytes;

    while (readAmount > 0)
    {
        // If readAmount is smaller than the remaining bytes
        // to be read in this frame, simply read it in and
        // we're done
        if (readAmount < frameReadAmount)
        {
            fread( dest, readAmount, 1, srcStream );
            break;
        }

        // Otherwise, read in the next frame
        fread( dest, frameReadAmount, 1, srcStream );

        // Reduce the read amount accordingly
        readAmount -= frameReadAmount;

        // We're now at the beginning of the next frame, so
        // consume one header's worth of data
        fread( &vhdr_dummy, VDIFHeaderBytes, 1, srcStream );
    }

    // Re-place the file position indicator
    fseek( srcStream, dataPosToFilePos( final_pos ), SEEK_SET );
}

void VDIFBuffer::setSrcFile( const char *srcFile, const char *mode )
{
    srcStream = fopen( srcFile, mode );
    if (srcStream == NULL)
        fileBytes = 0;
    else
    {
        // Measure the file length (but we'll subtract the
        // contribution by the headers below)
        fseek( srcStream, 0, SEEK_END );
        size_t rawFileBytes = ftell( srcStream );

        // Read in the VDIF header from the beginning of the file
        rewind( srcStream );
        fread( &vhdr, sizeof(vhdr), 1, srcStream );
        // The file position indicator is now pointing at the first
        // non-header byte

        // To work out the equivalent length of only the data,
        // We have to subtract the size of the header from each
        // frame
        VDIFFrameBytes  = getVDIFFrameBytes( &vhdr );
        VDIFHeaderBytes = getVDIFHeaderBytes( &vhdr );
        VDIFDataBytes   = VDIFFrameBytes - VDIFHeaderBytes;

        // Check that there is, indeed, a whole number of frames.
        // If there isn't, truncate and pretend that there is.
        if (rawFileBytes % VDIFFrameBytes != 0)
            rawFileBytes -= rawFileBytes % VDIFFrameBytes;

        // Calculate the number of frames...
        long nFrames = rawFileBytes / VDIFFrameBytes;

        // ...and the data size (which we'll hijack "fileBytes" with)
        fileBytes = nFrames * (VDIFFrameBytes - VDIFHeaderBytes);
    }
}

void VDIFBuffer::fillBuffer()
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
