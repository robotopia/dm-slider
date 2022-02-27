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

        //
        // Constructor/Destructor
        //

        VDIFBuffer( size_t bytes, const char *srcFile = NULL, const char *mode = "r" );
};

#endif
