#ifndef __DMSLIDER_H__
#define __DMSLIDER_H__

#include <glib.h>
#include <cuda_gl_interop.h>
#include <cufft.h>

#include "cohdd.h"

struct vdif_file
{
    char  *hdrfile;       // The name of the header file
    char  *hdr;           // The contents of the header file
    float  ctr_freq_MHz;
    float  bw_MHz;
    char  *datafile;      // The name of the data file
    void  *data;          // The contents of the data file (CPU)
    void  *d_data;        // The contents of the data file (GPU)
    uint32_t framelength; // The size of a frame (including header), in bytes
};

struct vdif_context
{
    GSList     *channels;
    size_t      nVDIFFrames;

    // Variables relating to dedispersion
    float       DM;
    int         taperType;

    uint32_t    Ns;
    uint32_t    Np;
    uint32_t    Nc;
    uint32_t    nframes;

    // Data buffers
    size_t          size;   // The size of the data (Np x Nc x Ns x sizeof(cuFloatComplex))
    cuFloatComplex *d_data; // An array containing cuFloatComplex data from all channels
    cuFloatComplex *d_spectrum;
    cuFloatComplex *d_dedispersed_spectrum;
    cuFloatComplex *d_dedispersed;

    // Global frequency metadata
    float       ref_freq_MHz;
    float       lo_freq_MHz;
    float       ctr_freq_MHz;
    float       hi_freq_MHz;
    float       bw_MHz;

    // cuFFT-related bookkeeping
    cufftHandle     plan;
};

// Defined in cohdd.cu:

void cudaCopyToSurface( cudaSurfaceObject_t surf, float *d_image, int w, int h );
void cudaCreateImage( float *d_image, int w, int h );

void cudaVDIFToFloatComplex( void *d_vds, void *d_vdif, size_t framelength, size_t headerlength,
        uint32_t Np, uint32_t Nc, uint32_t Ns, uint32_t c );

void cudaStokes( float *d_dest, cuFloatComplex *d_src, size_t NsNc, char stokes );

// Defined in vdif.c:

void init_vdif_context( struct vdif_context *vc, size_t nframes );

void load_vdif( struct vdif_file *vf, char *hdrfile, size_t nframes );

void destroy_all_vdif_files( struct vdif_context *vc );

void free_vdif_file( void * );

void add_vdif_file_to_context( void *, void * );
void add_vdif_files_to_context( struct vdif_context *vc, GSList *filenames );

void forwardFFT( struct vdif_context *vc );
void inverseFFT( struct vdif_context *vc );
void cudaScaleFactor( cuFloatComplex *d_data, float scale, size_t npoints );
void cudaCoherentDedispersion( cuFloatComplex *d_spectrum, cuFloatComplex *d_dedispersed_spectrum, size_t size,
        float DM, float ctr_freq_MHz_ch0, float ref_freq_MHz, float bw_MHz, int taperType, uint32_t Np, uint32_t Nc, uint32_t Ns );

#endif
