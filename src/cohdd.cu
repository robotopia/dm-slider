#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
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
 * @param in              Pointer to the input VDIF buffer
 * @param out             Pointer to the output buffer
 *                        (format: [Voltage dynamic spectrum](@ref arrayformats))
 * @param frameSizeBytes  The number of bytes per VDIF frame
 *                        (including the header)
 * @param headerSizeBytes The number of bytes per VDIF frame header
 *
 * Each thread converts one complex sample (= 2 bytes of input)
 * ```
 * <<< (Ns-1)/256+1, (256, Np) >>>
 * ```
 */
__global__ void cudaVDIFToFloatComplex_kernel( char2 *vdif, cuFloatComplex *vds, int frameSizeBytes, int headerSizeBytes, int Nc, int c )
{
    // The size of just the data part of the frame
    int dataSizeBytes = frameSizeBytes - headerSizeBytes;

    // It is assumed that `in` points to the first byte in a frameheader
    int s = threadIdx.x + blockIdx.x*blockDim.x; // Index of (non-header) data sample
    int p = threadIdx.y;

    int Ns = gridDim.x*blockDim.x;
    int Np = blockDim.y;

    // The input index (not counting frameheaders)
    int i = s*Np + p; // 2 bytes per sample

    // Get the frame number for this byte, and the idx within this frame
    int frame      = i / dataSizeBytes;
    int idxInFrame = i % dataSizeBytes;

    // Calculate the indices into the input and output arrays for this sample
    int vdif_idx = frame*frameSizeBytes + (headerSizeBytes + idxInFrame);
    int vds_idx  = p*Nc*Ns + c*Ns + s;

    // Bring the sample to register memory
    char2 v = vdif[vdif_idx];

    // Interpret each byte as an unsigned int
    uint8_t vx = *(uint8_t *)(&v.x);
    uint8_t vy = *(uint8_t *)(&v.y);

    // Turn them into floats and write it to global memory
    vds[vds_idx] = make_cuFloatComplex(
            ((float)vx)/256.0f - 0.5f,
            ((float)vy)/256.0f - 0.5f );
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
__global__ void cudaStokesI_kernel( cuFloatComplex *X, cuFloatComplex *Y, float *stokesI )
{
    // Let i represent the output sample index
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    // Pull out the two polarisations
    cuFloatComplex x = X[i];
    cuFloatComplex y = Y[i];

    // Calculate Stokes I
    stokesI[i] = x.x*x.x + x.y*x.y + y.x*y.x + y.y*y.y;
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

/*  ^      ^     ^
    |      |     |
   DEVICE FUNCTIONS
   ----------------
   HOST FUNCTIONS
    |     |     |
    v     v     v
*/

void cudaVDIFToFloatComplex( void *d_vds, void *d_vdif, size_t framelength, size_t headerlength,
        uint32_t Np, uint32_t Nc, uint32_t Ns, uint32_t c )
{
    dim3 blocks((Ns-1)/256+1);
    dim3 threads(256, Np);

    cudaVDIFToFloatComplex_kernel<<<blocks, threads>>>(
            (char2 *)d_vds,
            (cuFloatComplex *)d_vdif,
            framelength,
            headerlength,
            Nc, c );

    gpuErrchk( cudaDeviceSynchronize() );
}

void cudaStokesI( float *d_dest, cuFloatComplex *d_src, size_t NsNc )
{
    // Pull out the pointers to where the X and Y polarisations start
    cuFloatComplex *d_X = d_src;
    cuFloatComplex *d_Y = &d_src[NsNc];

    dim3 blocks((NsNc-1)/1024+1);
    dim3 threads(1024);

    cudaStokesI_kernel<<<blocks, threads>>>( d_X, d_Y, d_dest );
}

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


float *cudaCreateImage( float *d_image, int w, int h )
{
    cudaCreateImage_kernel<<<w,h>>>( d_image, w, h );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    return d_image;
}

