#ifndef __DMSLIDER_H__
#define __DMSLIDER_H__

#include <cuda_gl_interop.h>

struct vdif_file
{
    char  *hdrfile;       // The name of the header file
    char  *hdr;           // The contents of the header file
    float  ctr_freq_MHz;
    float  bw_MHz;
    void  *parent;
};

struct vdif_context
{
    unsigned int      nchannels;
    struct vdif_file *channels;
    float             ref_freq_MHz;
    float             DM;
    size_t            nframes;
    size_t            nsamples_max_view;
};

// Defined in cohdd.cu:

void cudaRotatePoints( float *points, float rad );

void cudaCopyToSurface( cudaSurfaceObject_t surf, float *d_image, int w, int h );

void cudaChangeBrightness( cudaSurfaceObject_t surf, float *d_image, float amount, int w, int h );

float *cudaCreateImage( cudaSurfaceObject_t surf, int w, int h );

// Defined in vdif.c:

//void load_vdifs...

#endif
