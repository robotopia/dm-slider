#ifndef __VDIFBUFFER_H__
#define __VDIFBUFFER_H__

#include <vdifio.h>
#include "SlideBuffer.h"

/**
 * VDIFBuffer class
 *
 * A child class of SlideBuffer, designed to handle the particular VDIF
 * format.
 */
class VDIFBuffer : public SlideBuffer
{
    private:

        long VDIFFrameBytes;  // The number of bytes per VDIF frame (including header)
        long VDIFHeaderBytes; // The number of bytes per VDIF header
        long VDIFDataBytes;   // The number of bytes per VDIF frame (excluding header)

       /**
         * Converts a position within a VDIF file to a position
         * within the VDIF data
         *
         * This function calculates what the given position within
         * a VDIF file *would* be if all the frame headers were
         * removed.
         *
         * If no VDIF file has been loaded, returns 0.
         *
         * @param pos A given "data" position within a VDIF file.
         * @return The equivalent "file" position.
         */
        long filePosToDataPos( long pos );

        /**
         * Converts a "data" position within a VDIF file to a
         * "file" position
         *
         * This function assumes that the given position is what the
         * position within a VDIF file *would* be if all the frame
         * headers were removed. It then calculates the true file
         * position, assuming 
         *
         * If no VDIF file has been loaded, returns 0.
         *
         * @param pos A given "data" position within a VDIF file.
         *            If a negative value is given, the current
         *            position of srcStream is used.
         * @return The equivalent position assuming no frame headers.
         */
        long dataPosToFilePos( long pos = -1 );

        /*
         * Get the current data position
         */
        inline size_t getCurrentDataPos() { return filePosToDataPos( ftell( srcStream ) ); }

    protected:

        vdif_header vhdr; // The VDIF header (defined in vdifio.h)

        /**
         * Utility function for reading bytes from a VDIF file stream
         *
         * The default implementation (Slidebuffer::readStreamToHost())
         * handles the case when the file stream contains contiguous memory,
         * which is not the case for VDIF files, which is organised into
         * "frames", each of which has a header.
         *
         * This implementation therefore needs to account for (i.e. discard)
         * the frame headers.
         */
        void readStreamToHost( long *slideAmount );

    public:

        /**
         * Set the source file stream
         *
         * As well as opening the file, this function also reads in the
         * first several bytes and interprets them as a VDIF header
         * struct.
         *
         * @param srcFile The name of the file to use as source
         * @param mode    The I/O mode (same as fopen())
         */
        void setSrcFile( const char *srcFile, const char *mode = "r" );

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
        void trimBuffer( size_t bytes = 0 );

        /**
         * Constructor for VDIFBuffer class
         *
         * This constructor allocates memory on the CPU and the GPU.
         *
         * @param bytes   The size of the slide buffer in bytes
         * @param srcFile The name of the file from which the data will be
         *                read
         * @param mode    The I/O mode (same as fopen())
         */
        VDIFBuffer( size_t bytes, const char *srcFile = NULL, const char *mode = "r" ) :
            SlideBuffer{ bytes, srcFile, mode, 2*bytes } {}

};

#endif
