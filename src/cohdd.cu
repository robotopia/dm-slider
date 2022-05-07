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
 * @param Nc              The number of channels
 * @param c               The channel number
 * @param Ns              The number of samples
 *
 * Each thread converts one complex sample (= 2 bytes of input)
 * ```
 * <<< (Ns-1)/512+1, (512, Np) >>>
 * ```
 */
__global__ void cudaVDIFToFloatComplex_kernel( char2 *vdif, cuFloatComplex *vds, int frameSizeBytes, int headerSizeBytes, int Nc, int c, int Ns )
{
    // Translate everything in terms of units of char2's:
    int frameSize  = frameSizeBytes/sizeof(char2);
    int headerSize = headerSizeBytes/sizeof(char2);
    int dataSize   = frameSize - headerSize;

    // It is assumed that `in` points to the first byte in a frameheader
    int s  = threadIdx.x + blockIdx.x*blockDim.x; // Index of (non-header) data sample
    int p  = threadIdx.y;
    int Np = blockDim.y;

    // Check for out-of-bounds
    if (s >= Ns)
        return;

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
 * @param N                           The number of spectral bins (=`Ns`)
 * @param DM                          The DM to apply (in pc/cm^3)
 * @param ctr_freq_MHz_ch0            The centre frequency of the 0th channel (MHz)
 * @param ref_freq_MHz                The reference frequency (MHz) for interchannel delays
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
        int N,
        float DM,
        float ctr_freq_MHz_ch0,
        float ref_freq_MHz,
        float bw_MHz,
        int taperType )
{
    //int Np = gridDim.x;                           // The number of polarisations
    int Nc = gridDim.y;                           // The number of channels

    int p = blockIdx.x;                           // The (p)olarisation number
    int c = blockIdx.y;                           // The (c)hannel number
    int n = blockIdx.z*blockDim.x + threadIdx.x;  // The spectral bin number

    // Check for out-of-bounds
    if (n >= N)
        return;

    int i = p*Nc*N + c*N + n;                     // The (i)ndex into both spectrum and dedispersed_spectrum

    float df = bw_MHz / (float)N;                 // Width of the spectral bin in MHz
    float f0 = ctr_freq_MHz_ch0 + bw_MHz*c;       // The centre frequency of the channel (MHz)
    float f  = (n <= N/2 ? df*n : df*(n-N));      // The freq of the spectral bin relative to f0 (MHz)
    float F  = f0 + f;                            // The absolute freq of the spectral bin (MHz)
    float dmphase = DMCONST*DM*f*f/(F*f0*f0);     // The dedispersion phase (s^-1 MHz^-1) to be applied to the spectral bin
    float dt = -calcTimeDelay( DM, ref_freq_MHz, f0 ); // The interchannel delay time
    float icphase = f*dt;                         // The interchannel delay phase (s^-1 MHz^-1)
    float Hr, Hi;                                 // The real and imag parts of H = exp(-2πi*phase)
    sincosf( -2.0f*CUDART_PI_F*1.0e6*(dmphase + icphase), &Hr, &Hi );
    float taper = taperFunc( f, bw_MHz, taperType ); // The taper function
    cuFloatComplex H = make_cuFloatComplex( Hr*taper, Hi*taper );

    dedispersed_spectrum[i] = cuCmulf( spectrum[i], H );
}

/**
 * Convert dual polarisation data to Stokes I
 *
 * @param X       The X-polarisation data
 * @param Y       The Y-polarisation data
 * @param stokesI The Stokes I output
 * @param N       The number of samples
 *
 * `data` is expected to be an array of *pairs* of complex numbers,
 * X,Y,X,Y,X,Y,...
 * from which the Stokes parameters are formed:
 *    I = |X|^2 + |Y|^2
 */
__global__ void cudaStokesI_kernel( cuFloatComplex *X, cuFloatComplex *Y, float *stokesI, int N )
{
    // Let i represent the output sample index
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= N)
        return;

    // Pull out the two polarisations
    cuFloatComplex x = X[i];
    cuFloatComplex y = Y[i];

    // Calculate Stokes I
    stokesI[i] = x.x*x.x + x.y*x.y + y.x*y.x + y.y*y.y;
}

/**
 * Convert dual polarisation data to Stokes Q
 *
 * @param X       The X-polarisation data
 * @param Y       The Y-polarisation data
 * @param stokesQ The Stokes Q output
 * @param N       The number of samples
 *
 * `data` is expected to be an array of *pairs* of complex numbers,
 * X,Y,X,Y,X,Y,...
 * from which the Stokes parameters are formed:
 *    Q = |X|^2 - |Y|^2
 */
__global__ void cudaStokesQ_kernel( cuFloatComplex *X, cuFloatComplex *Y, float *stokesQ, int N )
{
    // Let i represent the output sample index
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= N)
        return;

    // Pull out the two polarisations
    cuFloatComplex x = X[i];
    cuFloatComplex y = Y[i];

    // Calculate Stokes Q
    stokesQ[i] = x.x*x.x + x.y*x.y - y.x*y.x - y.y*y.y;
}

/**
 * Convert dual polarisation data to Stokes U
 *
 * @param X       The X-polarisation data
 * @param Y       The Y-polarisation data
 * @param stokesU The Stokes U output
 * @param N       The number of samples
 *
 * `data` is expected to be an array of *pairs* of complex numbers,
 * X,Y,X,Y,X,Y,...
 * from which the Stokes parameters are formed:
 *    U = 2*Re[X*Y]
 */
__global__ void cudaStokesU_kernel( cuFloatComplex *X, cuFloatComplex *Y, float *stokesU, int N )
{
    // Let i represent the output sample index
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= N)
        return;

    // Pull out the two polarisations
    cuFloatComplex x = X[i];
    cuFloatComplex y = Y[i];

    // Calculate Stokes U
    stokesU[i] = 2.0*(x.x*y.x - x.y*y.y);
}

/**
 * Convert dual polarisation data to Stokes V
 *
 * @param X       The X-polarisation data
 * @param Y       The Y-polarisation data
 * @param stokesV The Stokes V output
 * @param N       The number of samples
 *
 * `data` is expected to be an array of *pairs* of complex numbers,
 * X,Y,X,Y,X,Y,...
 * from which the Stokes parameters are formed:
 *    V = 2*Re[X*Y]
 */
__global__ void cudaStokesV_kernel( cuFloatComplex *X, cuFloatComplex *Y, float *stokesV, int N )
{
    // Let i represent the output sample index
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= N)
        return;

    // Pull out the two polarisations
    cuFloatComplex x = X[i];
    cuFloatComplex y = Y[i];

    // Calculate Stokes V
    stokesV[i] = -2.0*(x.x*y.y - x.y*y.x);
}

/**
 * Computes the mean power in consecutive bins
 *
 * @param[in]  unbinned  The original power dynamic spectrum
 * @param[out] binned    The time-scrunched power dynamic spectrum
 * @param      Ns        The size of the time dimension of `unbinned`
 * @param      Ns_binned The size of the time dimension of `binned`
 * @param      sfactor   The binning factor for samples
 * @param      Nc        The size of the frequency dimension of `unbinned`
 * @param      cfactor   The binning factor for channels
 *
 * Each thread in this non-optimised kernel sums `sfactor*cfactor` bins together.
 * If `factor` is small, it is hoped that the inefficiency of this
 * straightforward method is not too much worse than the overhead of
 * parallelising the sums.
 *
 * This kernel expects the thread/block layout to be:
 * ```
 * <<< ((B-1)/1024+1, C), 1024 >>>
 * ```
 * where `B` is the number of desired binned samples for this kernel to
 * compute, and `C` is the similarly number of desired binned channels.
 */
__global__ void cudaBinPower_kernel( float *unbinned, float *binned,
        int Ns, int Ns_binned, int sfactor, int Nc, int cfactor )
{
    int sb = blockIdx.x * blockDim.x + threadIdx.x; // The (s)ample number, in (b)inned
    int cb = blockIdx.y;                            // The (c)hannel number, in (b)inned
    int ib = cb*Ns_binned + sb;                     // The (i)ndex into (b)inned

    int s;                                          // The (s)ample number, in unbinned
    int c;                                          // The (c)hannel number, in unbinned
    int i;                                          // The (i)ndex into unbinned

    float res = 0.0;
    for (s = sb*sfactor; s < sb*sfactor + sfactor; s++)
    {
        for (c = cb*cfactor; c < cb*cfactor + cfactor; c++)
        {
            i = s*Ns + c;
            res += unbinned[i];
        }
    }

    // Put the mean into the binned array
    binned[ib] = res / (float)(sfactor*cfactor);
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
void cudaCopyToSurface_kernel( cudaSurfaceObject_t dest, float *src )
{
    int x = blockIdx.x;
    int y = threadIdx.x;
    int i = y*gridDim.x + x;

    surf2Dwrite( src[i], dest, x*sizeof(float), y );
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
    dim3 blocks((Ns-1)/512+1);
    dim3 threads(512, Np);

    cudaVDIFToFloatComplex_kernel<<<blocks, threads>>>(
            (char2 *)d_vdif,
            (cuFloatComplex *)d_vds,
            framelength,
            headerlength,
            Nc, c, Ns );

    gpuErrchk( cudaDeviceSynchronize() );
}

void cudaCoherentDedispersion( cuFloatComplex *d_spectrum, cuFloatComplex *d_dedispersed_spectrum, size_t size,
        float DM, float ctr_freq_MHz_ch0, float ref_freq_MHz, float bw_MHz, int taperType, uint32_t Np, uint32_t Nc, uint32_t Ns )
{
    dim3 blocks(Np, Nc, (Ns-1)/1024+1);
    dim3 threads(1024);

    cudaCoherentDedispersion_kernel<<<blocks, threads>>>(
            d_spectrum,
            d_dedispersed_spectrum,
            Ns,
            DM,
            ctr_freq_MHz_ch0,
            ref_freq_MHz,
            bw_MHz,
            taperType );

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cudaStokes( float *d_dest, cuFloatComplex *d_src, size_t NsNc, char stokes )
{
    // Pull out the pointers to where the X and Y polarisations start
    cuFloatComplex *d_X = d_src;
    cuFloatComplex *d_Y = &d_src[NsNc];

    dim3 blocks((NsNc-1)/1024+1);
    dim3 threads(1024);

    switch (stokes)
    {
        case 'I':
            cudaStokesI_kernel<<<blocks, threads>>>( d_X, d_Y, d_dest, NsNc );
            break;
        case 'Q':
            cudaStokesQ_kernel<<<blocks, threads>>>( d_X, d_Y, d_dest, NsNc );
            break;
        case 'U':
            cudaStokesU_kernel<<<blocks, threads>>>( d_X, d_Y, d_dest, NsNc );
            break;
        case 'V':
            cudaStokesV_kernel<<<blocks, threads>>>( d_X, d_Y, d_dest, NsNc );
            break;
        default:
            fprintf( stderr, "WARNING: Unrecognised Stokes parameter '%c'\n", stokes );
            break;
    }

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cudaCopyToSurface( cudaSurfaceObject_t surf, float *d_image, int w, int h )
{
    cudaCopyToSurface_kernel<<<w,h>>>( surf, d_image );
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
