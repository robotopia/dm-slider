#ifndef __CUDAERRORCHECKING_H__
#define __CUDAERRORCHECKING_H__

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;

// Boilerplate code for checking CUDA functions
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    /* Wrapper function for GPU/CUDA error handling. Every CUDA call goes through
       this function. It will return a message giving your the error string,
       file name and line of the error. Aborts on error. */

    if (code != 0)
    {
        fprintf(stderr, "GPUAssert:: %s - %s (%d)\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__, true);}

#endif
