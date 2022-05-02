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
    gpuErrchk( cudaFree( vds->d_data ) );
    gpuErrchk( cudaFree( vds->d_spectrum ) );
    gpuErrchk( cudaFree( vds->d_dedispersed_spectrum ) );
    gpuErrchk( cudaFree( vds->d_dedispersed ) );

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

    vds->size = NsNp * vds->Nc * sizeof(cuFloatComplex);

    gpuErrchk( cudaMalloc( (void **)&vds->d_data,                 vds->size ) );
    gpuErrchk( cudaMalloc( (void **)&vds->d_spectrum,             vds->size ) );
    gpuErrchk( cudaMalloc( (void **)&vds->d_dedispersed_spectrum, vds->size ) );
    gpuErrchk( cudaMalloc( (void **)&vds->d_dedispersed,          vds->size ) );

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
    if (cufftExecC2C( vds->plan, vds->d_data, vds->d_spectrum, CUFFT_FORWARD )
            != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: Could not execute (forward) cuFFT plan\n" );
    }
    gpuErrchk( cudaDeviceSynchronize() );
}

void inverseFFT( struct vds_t *vds )
{
    if (cufftExecC2C( vds->plan, vds->d_dedispersed_spectrum, vds->d_dedispersed, CUFFT_INVERSE )
            != CUFFT_SUCCESS)
    {
        fprintf( stderr, "WARNING: Could not execute (forward) cuFFT plan\n" );
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
