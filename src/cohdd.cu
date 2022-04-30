#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math_constants.h>
#include "cudaErrorChecking.h"

#include <stdlib.h>
#include <stdio.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_gl_interop.h>

#include "cohdd.h"

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
    // Translate everything in terms of units of char2's:
    int frameSize  = frameSizeBytes/sizeof(char2);
    int headerSize = headerSizeBytes/sizeof(char2);
    int dataSize   = frameSize - headerSize;

    // It is assumed that `in` points to the first byte in a frameheader
    int s = threadIdx.x + blockIdx.x*blockDim.x; // Index of (non-header) data sample
    int p = threadIdx.y;

    int Ns = gridDim.x*blockDim.x;
    int Np = blockDim.y;

    // The input index (not counting frameheaders)
    int i = s*Np + p; // 2 bytes per sample

    // Get the frame number for this byte, and the idx within this frame
    int frame      = i / dataSize;
    int idxInFrame = i % dataSize;

    // Calculate the indices into the input and output arrays for this sample
    int vdif_idx = frame*frameSize + (headerSize + idxInFrame);
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
 * Coherently dedisperse complex voltages
 *
 * @param[in] spectrum                The spectrum before dedispersion
 * @param[out] dediserpsed_spectrum   The spectrum after dedispersion
 * @param DM                          The DM to apply (in pc/cm^3)
 * @param ctr_freq_MHz_ch0            The centre frequency of the 0th channel (MHz)
 * @param bw_MHz                      The bandwidth of each channel (MHz)
 *
 * ```
 * <<< (Np, Nc, (Ns-1)/1024+1), 1024 >>>
 * ```
 * For now, this assumes that the channels are contiguous. Later I will consider
 * sparse channels (will likely have to put ctr freqs in shared memory)
 */
__global__
void cudaCoherentDedispersion_kernel(
        cuFloatComplex *spectrum,
        cuFloatComplex *dedispersed_spectrum,
        float DM,
        float ctr_freq_MHz_ch0,
        float bw_MHz)
{
    //int Np = gridDim.x;                           // The number of polarisations
    int Nc = gridDim.y;                           // The number of channels
    int N  = gridDim.z * blockDim.x;              // The number of spectral bins

    int p = blockIdx.x;                           // The (p)olarisation number
    int c = blockIdx.y;                           // The (c)hannel number
    int n = blockIdx.z*blockDim.x + threadIdx.x;  // The spectral bin number

    float df = bw_MHz / (float)N;                 // Width of the spectral bin in MHz
    float f0 = ctr_freq_MHz_ch0 + bw_MHz*c;       // The centre frequency of the channel (MHz)
    float f  = (n <= N/2 ? df*n : df*(n-N));      // The freq of the spectral bin relative to f0 (MHz)
    float F  = f0 + f;                            // The absolute freq of the spectral bin (MHz)
    float dmphase = -2.0f*CUDART_PI_F*1.0e6*DMCONST*DM*f*f/(F*f0*f0); // The phase (rad) to be applied to the spectral bin
    float Hr, Hi;                                 // The real and imag parts of H = exp(-2πi*dmphase)
    sincosf( dmphase, &Hr, &Hi );
    cuFloatComplex H = make_cuFloatComplex( Hr, Hi );

    int i = p*Nc*N + c*N + n;                     // The (i)ndex into both spectrum and dedispersed_spectrum
    dedispersed_spectrum[i] = cuCmulf( spectrum[i], H );
}

/**
 * Apply interchannel delays
 *
 * @param[in,out] spectrum       The spectrum after dedispersion
 * @param DM                     The DM to apply (in pc/cm^3)
 * @param ctr_freq_MHz_ch0       The centre frequency of the 0th channel (MHz)
 * @param bw_MHz                 The bandwidth of each channel (MHz)
 * @param ref_freq_MHz           The reference frequency (MHz)
 *
 * ```
 * <<< (Np, Nc, (Ns-1)/1024+1), 1024 >>>
 * ```
 * For now, this assumes that the channels are contiguous. Later I will consider
 * sparse channels (will likely have to put ctr freqs in shared memory)
 */
__global__
void cudaApplyInterchannelDelays_kernel(
        cuFloatComplex *spectrum,
        cuFloatComplex *dedispersed_spectrum,
        float DM,
        float ctr_freq_MHz_ch0,
        float bw_MHz,
        float ref_freq_MHz )
{
    //int Np = gridDim.x;                           // The number of polarisations
    int Nc = gridDim.y;                           // The number of channels
    int N  = gridDim.z * blockDim.x;              // The number of spectral bins

    int p = blockIdx.x;                           // The (p)olarisation number
    int c = blockIdx.y;                           // The (c)hannel number
    int n = blockIdx.z*blockDim.x + threadIdx.x;  // The spectral bin number

    float df = bw_MHz / (float)N;                 // Width of the spectral bin in MHz
    float f0 = ctr_freq_MHz_ch0 + bw_MHz*c;       // The centre frequency of the channel (MHz)
    float f  = (n <= N/2 ? df*n : df*(n-N));      // The freq of the spectral bin relative to f0 (MHz)
    float F  = f0 + f;                            // The absolute freq of the spectral bin (MHz)
    float dmphase = -2.0f*CUDART_PI_F*1.0e6*DMCONST*DM*f*f/(F*f0*f0); // The phase (rad) to be applied to the spectral bin
    float Hr, Hi;                                 // The real and imag parts of H = exp(-2πi*dmphase)
    sincosf( dmphase, &Hr, &Hi );
    cuFloatComplex H = make_cuFloatComplex( Hr, Hi );

    int i = p*Nc*N + c*N + n;                     // The (i)ndex into both spectrum and dedispersed_spectrum
    dedispersed_spectrum[i] = cuCmulf( spectrum[i], H );
}

/**
 * Apply a phase ramp to complex data
 *
 * @param data          The data to which the phase ramp is applied (in-place)
 * @param radPerBin     The slope of the phase ramp (in radians per bin)
 * @param samplesPerBin The number of contiguous samples to be rotated by the
 *                      same amount
 */
__global__
void cudaApplyPhaseRamp_kernel( cuFloatComplex *data, float radPerBin, int samplesPerBin )
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

__global__
void cudaScaleFactor_kernel( cuFloatComplex *data, float scale )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    data[i].x *= scale;
    data[i].y *= scale;
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
            (char2 *)d_vdif,
            (cuFloatComplex *)d_vds,
            framelength,
            headerlength,
            Nc, c );

    gpuErrchk( cudaDeviceSynchronize() );
}

void cudaCoherentDedispersion( cuFloatComplex *d_spectrum, cuFloatComplex *d_dedispersed_spectrum,
        float DM, float ctr_freq_MHz_ch0, float bw_MHz, uint32_t Np, uint32_t Nc, uint32_t Ns )
{
    dim3 blocks(Np, Nc, (Ns-1)/1024+1);
    dim3 threads(1024);

    cudaCoherentDedispersion_kernel<<<blocks, threads>>>(
            d_spectrum,
            d_dedispersed_spectrum,
            DM,
            ctr_freq_MHz_ch0,
            bw_MHz );

    gpuErrchk( cudaPeekAtLastError() );
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

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
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

void cudaScaleFactor( cuFloatComplex *d_data, float scale, size_t npoints )
{
    dim3 blocks((npoints-1)/1024+1);
    dim3 threads(1024);

    cudaScaleFactor_kernel<<<blocks, threads>>>( d_data, scale );

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
