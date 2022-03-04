#include <cuda_runtime.h>
#include "../src/cudaErrorChecking.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define diff_ns(START,END)  (1000000000L*(END.tv_sec - START.tv_sec) + END.tv_nsec - START.tv_nsec)

__global__ void cudaStripVDIFHeaders( char *in, char *out, int frameSizeBytes, int headerSizeBytes )
{
    // The size of just the data part of the frame
    int dataSizeBytes = frameSizeBytes - headerSizeBytes;

    // It is assumed that in points to the first byte in a frameheader
    int i = threadIdx.x + blockIdx.x*blockDim.x; // Index of (non-header) data byte

    // Get the frame number for this byte, and the idx within this frame
    int frame    = i/dataSizeBytes;
    int idxInFrame = i%dataSizeBytes;

    int in_idx  = frame*frameSizeBytes + (headerSizeBytes + idxInFrame);
    int out_idx = i;

    out[out_idx] = in[in_idx];
}

void strip_vdif_headers( char *in, char *out, int frameSizeBytes, int headerSizeBytes, int nFrames )
{
    // The size of just the data part of the frame
    int dataSizeBytes = frameSizeBytes - headerSizeBytes;

    int frame, idxInFrame;
    int in_idx, out_idx;

    for (frame = 0; frame < nFrames; frame++)
    {
        for (idxInFrame = 0; idxInFrame < dataSizeBytes; idxInFrame++)
        {
            out_idx = frame*dataSizeBytes + idxInFrame;
            in_idx  = frame*frameSizeBytes + (headerSizeBytes + idxInFrame);

            out[out_idx] = in[in_idx];
        }
    }
}

int main( int argc, char *argv[] )
{
    // Load VDIF data to the GPU two ways:
    //   1) By stripping out the frame headers on the CPU and loading a
    //      smaller amount of data to the GPU,
    //   2) By loading a larger amount of data onto the GPU and stripping
    //      out the frame headers there.
    //   Is there a scenario when #1 ever wins?

    // Create a large data array which will be our pretend VDIF file
    char *vdif, *d_vdif;
    char *vdif_no_header, *d_vdif_no_header;

    int nFrames = 10000;
    size_t frameSizeBytes  = 544;
    size_t headerSizeBytes = 32;
    size_t dataSizeBytes   = frameSizeBytes - headerSizeBytes;
    size_t vdifSizeBytes   = frameSizeBytes*nFrames;

    // Allocate memory
    gpuErrchk( cudaMallocHost( &vdif, vdifSizeBytes ) );
    gpuErrchk( cudaMallocHost( &vdif_no_header, vdifSizeBytes ) );
    gpuErrchk( cudaMalloc( &d_vdif, vdifSizeBytes ) );
    gpuErrchk( cudaMalloc( &d_vdif_no_header, vdifSizeBytes ) );

    int nFramesToRun;
    
    // Set up timing
    struct timespec start, end;
    printf( "# Number of frames | Time for method #1 | Time for method #2\n" );

    for (nFramesToRun = 0; nFramesToRun < nFrames; nFramesToRun++)
    {
        printf( "%d ", nFramesToRun );
        // (1) Strip the headers on the CPU
        /*
        clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start );
        {
            strip_vdif_headers(vdif, vdif_no_header, frameSizeBytes, headerSizeBytes, nFramesToRun);

            gpuErrchk(cudaMemcpy(d_vdif_no_header, vdif_no_header, nFramesToRun * dataSizeBytes, cudaMemcpyHostToDevice));
            gpuErrchk(cudaDeviceSynchronize());
        }
        clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &end );
        printf( "%ld ", diff_ns(start, end) );
        */

        // (2) Strip the headers on the GPU
        clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start );
        {
            gpuErrchk(cudaMemcpy(d_vdif, vdif, nFramesToRun * frameSizeBytes, cudaMemcpyHostToDevice));
            gpuErrchk(cudaDeviceSynchronize());

            dim3 blocks((nFramesToRun * dataSizeBytes) / 1024);
            dim3 threads(1024);
            cudaStripVDIFHeaders<<<blocks, threads>>>(d_vdif, d_vdif_no_header, frameSizeBytes, headerSizeBytes);
            gpuErrchk(cudaDeviceSynchronize());
        }
        clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &end );
        printf( "%ld\n", diff_ns(start, end) );
    }

    // Clean up memory
    gpuErrchk( cudaFree( d_vdif ) );
    gpuErrchk( cudaFree( d_vdif_no_header ) );
    gpuErrchk( cudaFreeHost( vdif ) );
    gpuErrchk( cudaFreeHost( vdif_no_header ) );

    // Exit gracefully
    return EXIT_SUCCESS;
}
