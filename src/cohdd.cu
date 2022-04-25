#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cudaErrorChecking.h"

#include <stdlib.h>
#include <stdio.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_gl_interop.h>

/**
 * Convert a VDIF buffer into an array of floats
 *
 * @param in              Pointer to the VDIF buffer
 * @param out             Pointer to the output buffer
 * @param frameSizeBytes  The number of bytes per VDIF frame
 *                        (including the header)
 * @param headerSizeBytes The number of bytes per VDIF frame header
 */
__global__ void cudaVDIFToFloatComplex_kernel( char2 *in, cuFloatComplex *out, int frameSizeBytes, int headerSizeBytes )
{
    // The size of just the data part of the frame
    int dataSizeBytes = frameSizeBytes - headerSizeBytes;

    // It is assumed that in points to the first byte in a frameheader
    int i = threadIdx.x + blockIdx.x*blockDim.x; // Index of (non-header) data sample

    // Express the index in terms of bytes
    int i2 = i*sizeof(char2);

    // Get the frame number for this byte, and the idx within this frame
    int frame      = i2 / dataSizeBytes;
    int idxInFrame = i2 % dataSizeBytes;

    // Calculate the indices into the input and output arrays for this sample
    int in_idx  = frame*frameSizeBytes + (headerSizeBytes + idxInFrame);
    int out_idx = i;

    // Bring the sample to register memory
    char2 sample = in[in_idx];

    // Turn it into a float and write it to global memory
    out[out_idx] = make_cuFloatComplex( (float)sample.x - 128.0, sample.y - 128.0 );
}

void cudaVDIFToFloatComplex( void *d_dest, void *d_src, size_t framelength, size_t headerlength, size_t nsamples )
{
    cudaVDIFToFloatComplex_kernel<<<nsamples/1024, 1024>>>(
                (char2 *)d_src,
                (cuFloatComplex *)d_dest,
                framelength,
                headerlength );
}

/**
 * Apply a phase ramp to complex data
 *
 * @param data          The data to which the phase ramp is applied (in-place)
 * @param radPerBin     The slope of the phase ramp (in radians per bin)
 * @param samplesPerBin The number of contiguous samples to be rotated by the
 *                      same amount
 */
__global__ void cudaApplyPhaseRamp_kernel( cuFloatComplex *data, float radPerBin, int samplesPerBin )
{
    // For this block/thread...
    int s = threadIdx.x + blockIdx.x*blockDim.x; // Get the (s)ample number
    int b = s / samplesPerBin;                   // Get the (b)in number

    // For each bin, calculate the phase rotation to be applied
    float rad = b * radPerBin;
    cuFloatComplex phase;
    sincosf( rad, &phase.y, &phase.x );

    // Apply the phase ramp (in-place)
    data[s] = cuCmulf( data[s], phase );
}

/**
 * Convert dual polarisation data to Stokes I
 *
 * @param data The data to be converted
 * @param stokesI The Stokes I output
 *
 * `data` is expected to be an array of *pairs* of complex numbers,
 * X,Y,X,Y,X,Y,...
 * from which the Stokes parameters are formed:
 *    I = |X|^2 + |Y|^2
 */
__global__ void cudaStokesI_kernel( cuFloatComplex *data, float *stokesI )
{
    // Let i represent the output sample index
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    // Pull out the two polarisations
    cuFloatComplex X = data[2*i];
    cuFloatComplex Y = data[2*i + 1];

    // Calculate Stokes I
    stokesI[i] = X.x*X.x + X.y*X.y + Y.x*Y.x + Y.y*Y.y;
}


__global__
void cudaCreateImage_kernel( float *image, int width, int height )
{
    // Makes an image of pixels ranging from 0.0 to 1.0, arranged in a gradient
    // so that top left is 0.0, bottom right is 1.0
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = i % width;
    int y = i / width;

    // Get normalised Manhattan distance
    float dist = (x + y)/(float)(width + height - 2);

    // Set the pixel value, with the peak being at the centre
    image[i] = dist;
}

__global__
void cudaCopyToSurface_kernel( cudaSurfaceObject_t dest, float *src, int width )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = i % width;
    int y = i / width;

    surf2Dwrite( src[i], dest, x*sizeof(float), y );
}

__global__
void cudaRotatePoints_kernel( float *points, float rad )
{
    // Assumes "points" is an array of sets of (x,y) coords
    // (i.e. two floats per point), with stride 4
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = 4;
    float x = points[stride*i];
    float y = points[stride*i+1];
    float s, c;
    sincosf( rad, &s, &c );
    points[stride*i]   = c*x - s*y;
    points[stride*i+1] = s*x + c*y;
}

__global__
void cudaChangeBrightness_kernel( float *image, float amount )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    image[i] += amount;
}

/*  ^      ^     ^
    |      |     |
   DEVICE FUNCTIONS
   ----------------
   HOST FUNCTIONS
    |     |     |
    v     v     v
*/

void cudaRotatePoints( float *d_points, float rad )
{
    cudaRotatePoints_kernel<<<1,4>>>( d_points, rad );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cudaCopyToSurface( cudaSurfaceObject_t surf, float *d_image, int w, int h )
{
    cudaCopyToSurface_kernel<<<w,h>>>( surf, d_image, w );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


void cudaChangeBrightness( cudaSurfaceObject_t surf, float *d_image, float amount, int w, int h )
{
    cudaChangeBrightness_kernel<<<w,h>>>( d_image, amount );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaCopyToSurface( surf, d_image, w, h );
}

float *cudaCreateImage( float *d_image, cudaSurfaceObject_t surf, int w, int h )
{
    cudaCreateImage_kernel<<<w,h>>>( d_image, w, h );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Copy it to the surface (which is mapped to the OpenGL texture)
    cudaCopyToSurface( surf, d_image, w, h );

    return d_image;
}

