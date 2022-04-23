#ifndef __DMSLIDER_H__
#define __DMSLIDER_H__

#include <cuda_gl_interop.h>

void cudaRotatePoints( float *points, float rad );

void cudaCopyToSurface( cudaSurfaceObject_t surf, float *d_image, int w, int h );

void cudaChangeBrightness( cudaSurfaceObject_t surf, float *d_image, float amount, int w, int h );

float *cudaCreateImage( cudaSurfaceObject_t surf, int w, int h );

#endif
