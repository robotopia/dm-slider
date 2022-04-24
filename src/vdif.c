#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>

#include <vdifio.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cudaErrorChecking.h"

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

inline void init_vdif_file( struct vdif_file *vf )
{
    memset( vf, 0, sizeof(struct vdif_file) );

    // Allocate memory just for the datafile name (since ascii_header.c
    // assumes it's already allocated)

    // Assume that data filenames will not be bigger than 4096 bytes long
    vf->datafile = (char *)malloc( 4096 );

}

void add_vdif_file_to_context( void *ptr1, void *ptr2 )
{
    char *hdrfile = (char *)ptr1;
    struct vdif_context *vc = (struct vdif_context *)ptr2;

    struct vdif_file *vf = (struct vdif_file *)malloc( sizeof(struct vdif_file) );
    init_vdif_file( vf );
    load_vdif( vf, hdrfile, vc->nframes );
    vc->channels = g_slist_append( vc->channels, vf );
}

void add_vdif_files_to_context( struct vdif_context *vc, GSList *filenames )
{
    g_slist_foreach( filenames, add_vdif_file_to_context, vc );
}

void load_vdif( struct vdif_file *vf, char *hdrfile, size_t nframes )
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

    char datafile[4096];
    // Parse frequency information
    ascii_header_get( vf->hdr, "FREQ",     "%f", &vf->ctr_freq_MHz );
    ascii_header_get( vf->hdr, "BW",       "%f", &vf->bw_MHz       );
    ascii_header_get( vf->hdr, "DATAFILE", "%s", datafile          );

    // Prepend the same path from hdrfile to datafile
    char *s = strrchr( hdrfile, '/' );
    if (!s)
    {
        // If no path was found, don't prepend one!
        strcpy( vf->datafile, datafile );
    }
    else
    {
        // Some pointer fun to copy the path, and then append the filename
        int n = s - hdrfile + 1;
        strncpy( vf->datafile, hdrfile, s-hdrfile+1 );
        s = vf->datafile + n;
        strcpy( s, datafile );
    }

    // Open the datafile and read in the first header to get the size of a frame
    FILE *f = fopen( vf->datafile, "r" );
    if (!f)
    {
        fprintf( stderr, "WARNING: could not open %s\n", vf->datafile );
        return;
    }

    vdif_header vhdr;
    fread( &vhdr, VDIF_HEADER_BYTES, 1, f );
    vf->framelength = vhdr.framelength8 * 8;

    // Allocate memory on both CPU and GPU
    size_t nbytes = vf->framelength * nframes;
    gpuErrchk( cudaMallocHost( &vf->data, nbytes ) );
    gpuErrchk( cudaMalloc( &vf->d_data, nbytes ) );

    // Read in the data
    rewind( f );
    fread( vf->data, nbytes, 1, f );
    fclose( f );

    // Upload to the GPU
    gpuErrchk( cudaMemcpy( vf->d_data, vf->data, nbytes, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaDeviceSynchronize() );
}

void free_vdif_file( void *ptr )
{
    if (!ptr)
        return;

    struct vdif_file *vf = (struct vdif_file *)ptr;

    free( vf->hdrfile );
    free( vf->hdr );
    free( vf->datafile );
    cudaFreeHost( vf->data );
    cudaFree( vf->d_data );
    free( vf );
}

void destroy_all_vdif_files( struct vdif_context *vc )
{
    if (!vc)
        return;

    g_slist_free_full( vc->channels, free_vdif_file );
}
