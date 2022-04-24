#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <vdifio.h>

#include "dm-slider.h"
#include "ascii_header.h"

void init_vdif_context( struct vdif_context *vc, size_t nframes, size_t nsamples_max_view )
{
    vc->nchannels           = 0;
    vc->channels            = NULL;
    vc->ref_freq_MHz        = 0.0;
    vc->DM                  = 0.0;
    vc->nframes             = nframes;
    vc->nsamples_max_view   = nsamples_max_view;
}

void load_vdif( struct vdif_file *vf, char *hdrfile )
{
    // Check that non-null pointers were given
    if (!vf)
    {
        fprintf( stderr, "WARNING: invalid pointer to vdif_file struct. Nothing loaded.\n" );
        return;
    }

    if (!hdrfile)
    {
        fprintf( stderr, "WARNING: invalid pointer to vdif hdr file name. Nothing loaded.\n" );
        return;
    }

    // Set the hdrfile name
    vf->hdrfile = (char *)malloc( strlen(hdrfile) + 1 );
    strcpy( vf->hdrfile, hdrfile );

    // Load the header file contents
    FILE *fhdr = fopen( hdrfile, "r" );
    if (!fhdr)
    {
        fprintf( stderr, "WARNING: '%s' unable to be opened.\n", hdrfile );
        return;
    }

    fseek( fhdr, 0L, SEEK_END );
    size_t nbytes = ftell( fhdr );
    rewind( fhdr );

    vf->hdr = (char *)malloc( nbytes );
    fread( vf->hdr, nbytes, 1, fhdr );

    // Parse frequency information
    ascii_header_get( vf->hdr, "FREQ", "%f", &vf->ctr_freq_MHz );
    ascii_header_get( vf->hdr, "BW",   "%f", &vf->bw_MHz       );

    // Close file
    fclose( fhdr );
}
