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
#include "title.h"

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

    vds_malloc_gpu( vds, vds->Ns * vds->Np * vds->Nc * sizeof(cuFloatComplex) );
    vds->ref_freq_MHz = 150.0;
    vds->lo_freq_MHz  = 100.0;
    vds->ctr_freq_MHz = 150.0;
    vds->hi_freq_MHz  = 200.0;
    vds->bw_MHz       = 100.0;

    uint8_t Xdata[] = TITLE_IMAGE; // Defined in title.h

    cuFloatComplex data[vds->Ns * vds->Nc * vds->Np];

    uint32_t c, s, Xi, Yi, i;
    for (c = 0; c < vds->Nc; c++)
    {
        for (s = 0; s < vds->Ns; s++)
        {
            i = c*vds->Ns + s;
            Xi = i;
            Yi = i + 1*vds->Nc*vds->Ns;
            data[Xi] = make_cuFloatComplex( float(Xdata[i])/255.0, 0.0 );
            data[Yi] = make_cuFloatComplex( 0.0, 0.0 );
        }
    }

    gpuErrchk( cudaMemcpy( vds->d_data, data, vds->size, cudaMemcpyHostToDevice ) );
}

void vds_from_vdif_context( struct vds_t *vds, struct vdif_context *vc )
{
    // Pull out some of the (frequency) metadata for the whole set of channels
    struct vdif_file *vf = (struct vdif_file *)vc->channels->data; // The first channel
    vds->lo_freq_MHz      = vf->ctr_freq_MHz - vf->bw_MHz/2.0;
    vf                   = (struct vdif_file *)g_slist_last( vc->channels )->data; // The last channel
    vds->hi_freq_MHz      = vf->ctr_freq_MHz + vf->bw_MHz/2.0;
    vds->ctr_freq_MHz     = 0.5*(vds->lo_freq_MHz + vds->hi_freq_MHz);
    vds->bw_MHz           = vds->hi_freq_MHz - vds->lo_freq_MHz;
    vds->ref_freq_MHz     = vds->ctr_freq_MHz;

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
