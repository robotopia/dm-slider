#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>

#include <vdifio.h>

#include "dm-slider.h"
#include "ascii_header.h"

void init_vdif_context( struct vdif_context *vc, size_t nframes, size_t nsamples_max_view )
{
    vc->channels            = NULL;
    vc->ref_freq_MHz        = 0.0;
    vc->DM                  = 0.0;
    vc->nframes             = nframes;
    vc->nsamples_max_view   = nsamples_max_view;
}

void add_vdif_file_to_context( void *ptr1, void *ptr2 )
{
    char *hdrfile = (char *)ptr1;
    struct vdif_context *vc = (struct vdif_context *)ptr2;

    struct vdif_file *vf = (struct vdif_file *)malloc( sizeof(struct vdif_file) );
    load_vdif( vf, hdrfile );
    vc->channels = g_slist_append( vc->channels, vf );
}

void add_vdif_files_to_context( struct vdif_context *vc, GSList *filenames )
{
    g_slist_foreach( filenames, add_vdif_file_to_context, vc );
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
    vf->hdr = load_file_contents_as_str( hdrfile );

    // Parse frequency information
    ascii_header_get( vf->hdr, "FREQ", "%f", &vf->ctr_freq_MHz );
    ascii_header_get( vf->hdr, "BW",   "%f", &vf->bw_MHz       );
}

void free_vdif_file( void *ptr )
{
    if (!ptr)
        return;

    struct vdif_file *vf = (struct vdif_file *)ptr;
    free( vf->hdrfile );
    free( vf->hdr );
    free( vf );
}

void destroy_all_vdif_files( struct vdif_context *vc )
{
    if (!vc)
        return;

    g_slist_free_full( vc->channels, free_vdif_file );
}
