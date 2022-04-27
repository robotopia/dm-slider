#ifndef __DMSLIDER_H__
#define __DMSLIDER_H__

#include <glib.h>
#include <cuda_gl_interop.h>

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
    float       DM;
    size_t      nframes;
    size_t      nsamples_max_view;
    uint32_t    npols;
    size_t      ndual_pol_samples;
    size_t      size;       // The size of the loaded data from all channels as cuFloatComplex (d_data)
    cuFloatComplex *d_data; // An array containing cuFloatComplex data from all channels
    float       ref_freq_MHz;
    float       lo_freq_MHz;
    float       ctr_freq_MHz;
    float       hi_freq_MHz;
    float       bw_MHz;
};

// Defined in cohdd.cu:

void cudaRotatePoints( float *points, float rad );
void cudaCopyToSurface( cudaSurfaceObject_t surf, float *d_image, int w, int h );
void cudaCreateImage( float *d_image, int w, int h );

void cudaVDIFToFloatComplex( void *d_dest, void *d_src, size_t framelength, size_t headerlength, size_t nsamples );
void cudaStokesI( float *d_dest, cuFloatComplex *d_src, size_t nDualPolSamples );

// Defined in vdif.c:

void init_vdif_context( struct vdif_context *vc, size_t nframes, size_t nsamples_max_view );

void load_vdif( struct vdif_file *vf, char *hdrfile, size_t nframes );

void destroy_all_vdif_files( struct vdif_context *vc );

void free_vdif_file( void * );

void add_vdif_file_to_context( void *, void * );
void add_vdif_files_to_context( struct vdif_context *vc, GSList *filenames );

#endif
