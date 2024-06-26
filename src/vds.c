#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include "cudaErrorChecking.h"

#include <vdifio.h>

#include "dm-slider.h"

void vds_init( struct vds_t *vds )
{
    memset( vds, 0, sizeof(struct vds_t) );
}

void vds_malloc_gpu( struct vds_t *vds, size_t size )
{
    vds->size = size;

    gpuErrchk( cudaMalloc( (void **)&vds->d_data,                 vds->size ) );
    gpuErrchk( cudaMalloc( (void **)&vds->d_spectrum,             vds->size ) );
    gpuErrchk( cudaMalloc( (void **)&vds->d_dedispersed_spectrum, vds->size ) );
    gpuErrchk( cudaMalloc( (void **)&vds->d_dedispersed,          vds->size ) );
}

void vds_free_gpu( struct vds_t *vds )
{
    gpuErrchk( cudaFree( vds->d_data ) );
    gpuErrchk( cudaFree( vds->d_spectrum ) );
    gpuErrchk( cudaFree( vds->d_dedispersed_spectrum ) );
    gpuErrchk( cudaFree( vds->d_dedispersed ) );

    vds->d_data                 = NULL;
    vds->d_spectrum             = NULL;
    vds->d_dedispersed_spectrum = NULL;
    vds->d_dedispersed          = NULL;

    vds->size = 0;
}

void vds_create_title( struct vds_t *vds )
{
    vds_free_gpu( vds );
    vds_init( vds );

    vds->Ns = 512;
    vds->Np = 2;
    vds->Nc = 128;

    vds->DM = 0.0;

    vds_malloc_gpu( vds, vds->Ns * vds->Np * vds->Nc * sizeof(cuFloatComplex) );
    vds_set_freq_ctr_bw( vds, 150.0, 100.0 );
    vds->ref_freq_MHz = 150.0;

    vds->dt = 1.0e-6/channel_bw_MHz( vds ); // sec

#ifndef RESOURCE_FOLDER
    FILE *f = fopen( "res/title.dat", "r" );
#else
    char title_filename[1024];
    sprintf( title_filename, "%s/title.dat", RESOURCE_FOLDER );
    FILE *f = fopen( title_filename, "r" );
#endif
    fseek( f, 0L, SEEK_END );
    size_t title_size = ftell( f );
    rewind( f );

    uint8_t Xdata[title_size];
    fread( Xdata, title_size, 1, f );
    fclose( f );

    cuFloatComplex data[vds->Ns * vds->Nc * vds->Np];

    uint32_t c, s, Xi, Yi, i;
    for (c = 0; c < vds->Nc; c++)
    {
        for (s = 0; s < vds->Ns; s++)
        {
            i = (vds->Nc - c - 1)*vds->Ns + s;
            Xi = c*vds->Ns + s;
            Yi = Xi + 1*vds->Nc*vds->Ns;
            data[Xi] = make_cuFloatComplex( float(Xdata[i])/255.0, 0.0 );
            data[Yi] = make_cuFloatComplex( 0.0, 0.0 );
        }
    }

    // Copy data to GPU
    gpuErrchk( cudaMemcpy( vds->d_data, data, vds->size, cudaMemcpyHostToDevice ) );

    // Initialise the spectrum
    vds_spectrum_init( vds );
}

void vds_set_freq_ctr_bw( struct vds_t *vds, float ctr_freq_MHz, float bw_MHz )
{
    vds->ctr_freq_MHz = ctr_freq_MHz;
    vds->bw_MHz       = bw_MHz;
    vds->lo_freq_MHz  = ctr_freq_MHz - bw_MHz/2.0;
    vds->hi_freq_MHz  = ctr_freq_MHz + bw_MHz/2.0;
}

void vds_set_freq_lo_hi( struct vds_t *vds, float lo_freq_MHz, float hi_freq_MHz )
{
    vds->lo_freq_MHz  = lo_freq_MHz;
    vds->hi_freq_MHz  = hi_freq_MHz;
    vds->ctr_freq_MHz = 0.5*(lo_freq_MHz + hi_freq_MHz);
    vds->bw_MHz       = hi_freq_MHz - lo_freq_MHz;
}

void vds_from_vdif_context( struct vds_t *vds, struct vdif_context *vc )
{
    // Derive global frequency info from VDIF channels
    struct vdif_file *vf = (struct vdif_file *)vc->channels->data; // The first channel
    float lo_freq_MHz    = vf->ctr_freq_MHz - vf->bw_MHz/2.0; // Bottom edge of bottom channel
    vf                   = (struct vdif_file *)g_slist_last( vc->channels )->data; // The last channel
    float hi_freq_MHz    = vf->ctr_freq_MHz + vf->bw_MHz/2.0; // op edge of top channel
    vds_set_freq_lo_hi( vds, lo_freq_MHz, hi_freq_MHz );

    vds->dt              = vc->dt;
    vds->DM              = 0.0;

    // Only after they're all loaded, allocate a GPU array which will house
    // all the VDIF data (i.e. from all the channels) in cuComplex form.

    // Remove any previous allocation, if any
    vds_free_gpu( vds );

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
    vds->Np = 2;
    vds->Ns = NsNp / vds->Np;
    vds->Nc = g_slist_length( vc->channels );

    vds_malloc_gpu( vds, NsNp * vds->Nc * sizeof(cuFloatComplex) );

    // Run the kernels to convert from VDIF bytes to cuFloatComplex
    i = vc->channels;
    uint32_t c;
    for (c = 0; c < vds->Nc; c++)
    {
        vf = (struct vdif_file *)i->data;

        cudaVDIFToFloatComplex( vds->d_data, vf->d_data, framelength, VDIF_HEADER_BYTES,
                vds->Np, vds->Nc, vds->Ns, c );

        i = i->next;
    }

    // Initialise the spectrum
    vds_spectrum_init( vds );
}

void vds_spectrum_init( struct vds_t *vds )
{
    // Set up cuFFT plan
    // Because the arrays have time as the fastest changing quantity, the plan
    // can be a batch (1D) plan that does all channels and polarisations
    // at once
    int n = (int)vds->Ns;
    if (cufftPlanMany(
                &vds->plan,       // Where the plan, once created, resides
                1,                // The number of dimensions
                &n,               // The size of each transform
                NULL,             // Input storage dimenions (NULL means use default = packed)
                0, 0,             // Stride parameters ignored if previous parameter is NULL
                NULL,             // Output storage dimensions (NULL means use default = packed)
                0, 0,             // Stride parameters ignored if previous parameter is NULL
                CUFFT_C2C,        // Transform type (C2C = complex-to-complex single precision)
                vds->Nc * vds->Np // The batch size
                )
            != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: Could not create cuFFT plan\n" );
    }

    // FFT everything straight away
    forwardFFT( vds );
}

void forwardFFT( struct vds_t *vds )
{
    cufftResult res = cufftExecC2C( vds->plan, vds->d_data, vds->d_spectrum, CUFFT_FORWARD );
    if (res != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: Could not execute (forward) cuFFT plan (error code = %d)\n", res );
    }
    gpuErrchk( cudaDeviceSynchronize() );
}

void inverseFFT( struct vds_t *vds )
{
    cufftResult res = cufftExecC2C( vds->plan, vds->d_dedispersed_spectrum, vds->d_dedispersed, CUFFT_INVERSE );
    if (res != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: Could not execute (inverse) cuFFT plan (error code = %d)\n", res );
    }
    gpuErrchk( cudaDeviceSynchronize() );

    cudaScaleFactor( vds->d_dedispersed, 1.0/vds->Ns, vds->Np * vds->Nc * vds->Ns );
    gpuErrchk( cudaDeviceSynchronize() );
}

void vds_destroy( struct vds_t *vds )
{
    if (!vds)
        return;

    // Destroy cuFFT plan
    if (cufftDestroy( vds->plan ) != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: could not destroy cuFFT plan\n" );
    }
}

float channel_bw_MHz( struct vds_t *vds )
{
    return vds->bw_MHz / (float)vds->Nc;
}

float ctr_freq_MHz_nth_channel( struct vds_t *vds, uint32_t n )
{
    return (n + 0.5)*channel_bw_MHz( vds ) + vds->lo_freq_MHz;
}
