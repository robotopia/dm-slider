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
    float       ref_freq_MHz;
    float       DM;
    size_t      nframes;
    size_t      nsamples_max_view;
};

// Defined in cohdd.cu:

void cudaRotatePoints( float *points, float rad );

void cudaCopyToSurface( cudaSurfaceObject_t surf, float *d_image, int w, int h );

void cudaChangeBrightness( cudaSurfaceObject_t surf, float *d_image, float amount, int w, int h );

float *cudaCreateImage( cudaSurfaceObject_t surf, int w, int h );

// Defined in vdif.c:

void init_vdif_context( struct vdif_context *vc, size_t nframes, size_t nsamples_max_view );

void load_vdif( struct vdif_file *vf, char *hdrfile, size_t nframes );

void destroy_all_vdif_files( struct vdif_context *vc );

void free_vdif_file( void * );

void add_vdif_file_to_context( void *, void * );
void add_vdif_files_to_context( struct vdif_context *vc, GSList *filenames );

#endif
