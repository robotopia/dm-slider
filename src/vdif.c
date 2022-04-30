#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>

#include <vdifio.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include "cudaErrorChecking.h"

#include "dm-slider.h"
#include "ascii_header.h"

void init_vdif_context( struct vdif_context *vc, size_t nframes )
{
    vc->channels            = NULL;
    vc->ref_freq_MHz        = 0.0;
    vc->DM                  = 0.0;
    vc->nframes             = nframes;
    vc->size                = 0;
    vc->d_data              = NULL;
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
    // Add each file, one at a time
    g_slist_foreach( filenames, add_vdif_file_to_context, vc );

    // Pull out some of the (frequency) metadata for the whole set of channels
    struct vdif_file *vf = (struct vdif_file *)vc->channels->data; // The first channel
    vc->lo_freq_MHz      = vf->ctr_freq_MHz - vf->bw_MHz/2.0;
    vf                   = (struct vdif_file *)g_slist_last( vc->channels )->data; // The last channel
    vc->hi_freq_MHz      = vf->ctr_freq_MHz + vf->bw_MHz/2.0;
    vc->ctr_freq_MHz     = 0.5*(vc->lo_freq_MHz + vc->hi_freq_MHz);
    vc->bw_MHz           = vc->hi_freq_MHz - vc->lo_freq_MHz;
    vc->ref_freq_MHz     = vc->ctr_freq_MHz;

    // Only after they're all loaded, allocate a GPU array which will house
    // all the VDIF data (i.e. from all the channels) in cuComplex form.

    // Remove any previous allocation, if any
    gpuErrchk( cudaFree( vc->d_data ) );
    gpuErrchk( cudaFree( vc->d_spectrum ) );
    gpuErrchk( cudaFree( vc->d_dedispersed_spectrum ) );
    gpuErrchk( cudaFree( vc->d_dedispersed ) );

    // Check that all channels have the same framelength
    bool all_same = true;
    vf = (struct vdif_file *)vc->channels->data; // The first channel
    uint32_t framelength = vf->framelength; // The reference framelength
    GSList *i;
    for (i = vc->channels; i != NULL; i = i->next)
    {
        vf = (struct vdif_file *)i->data;
        if (vf->framelength != framelength)
        {
            all_same = false;
            break;
        }
    }

    if (!all_same)
    {
        fprintf( stderr, "WARNING: the selected VDIF files have different "
                "framelengths. Complex data array not initialised\n" );
        return;
    }

    // Allocate memory:
    //   Np = Number of polarisations
    //   Nc = Number of (frequency) channels
    //   Ns = Number of (time) samples
    size_t bytes_per_chan = vc->nframes * (framelength - VDIF_HEADER_BYTES);
    size_t NsNp = bytes_per_chan / 2; // 2 = ncmplx
                                      //
    vc->Np = 2;
    vc->Ns = NsNp / vc->Np;
    vc->Nc = g_slist_length( vc->channels );

    vc->size = NsNp * vc->Nc * sizeof(cuFloatComplex);

    gpuErrchk( cudaMalloc( (void **)&vc->d_data,                 vc->size ) );
    gpuErrchk( cudaMalloc( (void **)&vc->d_spectrum,             vc->size ) );
    gpuErrchk( cudaMalloc( (void **)&vc->d_dedispersed_spectrum, vc->size ) );
    gpuErrchk( cudaMalloc( (void **)&vc->d_dedispersed,          vc->size ) );

    // Run the kernels to convert from VDIF bytes to cuFloatComplex
    i = vc->channels;
    uint32_t c;
    for (c = 0; c < vc->Nc; c++)
    {
        vf = (struct vdif_file *)i->data;

        cudaVDIFToFloatComplex( vc->d_data, vf->d_data, framelength, VDIF_HEADER_BYTES,
                vc->Np, vc->Nc, vc->Ns, c );

        i = i->next;
    }

    // Set up cuFFT plan
    // Because the arrays have time as the fastest changing quantity, the plan
    // can be a batch (1D) plan that does all channels and polarisations
    // at once
    int n = (int)vc->Ns;
    if (cufftPlanMany(
                &vc->plan,      // Where the plan, once created, resides
                1,              // The number of dimensions
                &n,             // The size of each transform
                NULL,           // Input storage dimenions (NULL means use default = packed)
                0, 0,           // Stride parameters ignored if previous parameter is NULL
                NULL,           // Output storage dimensions (NULL means use default = packed)
                0, 0,           // Stride parameters ignored if previous parameter is NULL
                CUFFT_C2C,      // Transform type (C2C = complex-to-complex single precision)
                vc->Nc * vc->Np // The batch size
                )
            != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: Could not create cuFFT plan\n" );
    }

    // FFT everything straight away
    forwardFFT( vc );
}

void forwardFFT( struct vdif_context *vc )
{
    if (cufftExecC2C( vc->plan, vc->d_data, vc->d_spectrum, CUFFT_FORWARD )
            != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: Could not execute (forward) cuFFT plan\n" );
    }
    gpuErrchk( cudaDeviceSynchronize() );
}

void inverseFFT( struct vdif_context *vc )
{
    if (cufftExecC2C( vc->plan, vc->d_dedispersed_spectrum, vc->d_dedispersed, CUFFT_INVERSE )
            != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: Could not execute (forward) cuFFT plan\n" );
    }
    gpuErrchk( cudaDeviceSynchronize() );

    cudaScaleFactor( vc->d_dedispersed, 1.0/vc->Ns, vc->Np * vc->Nc * vc->Ns );
    gpuErrchk( cudaDeviceSynchronize() );
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

    // Destroy cuFFT plan
    if (cufftDestroy( vc->plan ) != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: could not destroy cuFFT plan\n" );
    }
}
