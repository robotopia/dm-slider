# Data pipeline

\dotfile cudaopengl.dot

| Process | Array format | Associated function(s) | Size (1 second,<br>24 x 1.28 MHz channels) |
| :------ | :----- | :---------- | :------------ |
| **START** | VDIF, or other format | `cudaMemcpy()` | 125 MB (GPU)<br>(3 seconds, 24 coarse channels) |
| Convert to VDS | [Voltage dynamic spectrum](@ref arrayformats) | `vds_create_title()`<br>`vds_from_vdif_context()`<br>`cudaVDIFToFloatComplex()` | 500 MB |
| Fourier transform | [Voltage dynamic spectrum](@ref arrayformats) | `forwardFFT()` | 500 MB |
| Coherently dedisperse<br>channels | [Voltage dynamic spectrum](@ref arrayformats) | `cudaCoherentDedispersion()` | 500 MB |
| Inverse Fourier transform | [Voltage dynamic spectrum](@ref arrayformats) | `inverseFFT()` | 500 MB |
| Form Stokes (I,Q,U,V) | [Power dynamic spectrum](@ref arrayformats) | `cudaStokes()` | 125 MB |
| Binning | [Power dynamic spectrum](@ref arrayformats) | `cudaBinPower()` |  |
| Copy to CUDA Surface | [Image](@ref arrayformats) | `cudaCopyToSurface()` |  |
| Map to OpenGL Texture | [Image](@ref arrayformats) | CUDA-OpenGL interoperability |  |
| Draw on quad (via shaders) |  |  |  |

\dotfile datapipeline.dot

In general, the data volume is too big for *all* the data to process at once.
At each step along the processing pipeline, choices must be made as to which subset of data should be processed.
These decisions are based on the primary design paradigm of this piece of software: **smooth interactivity**.

In particular, the three primary mode of (mouse-driven) interaction are:

1. Scroll [*left click and drag*]
2. Zoom [*mouse scroll*]
3. Change dispersion measure [*right click and drag*]

In general, #1 and #2 can be entirely handled by OpenGL, unless the user scrolls beyond the edge of the currently loaded subset of data.
Changing the dispersion measure, however, is necessarily handled further upstream by CUDA, and is therefore necessarily more expensive.
The question is, how much data should be dedispersed in one go?
On one hand, as small a subset as necessary will ensure that the interactive dedispersion can run as smoothly as possibly.
On the other hand, dedispersing a larger subset at once will avoid having to re-dedisperse when the data are scrolled and zoomed.
Another limitation is the maximum allowed size for textures in OpenGL.

## Array formats {#arrayformats}

| Format | Dimensions (from slowest to<br>fastest changing) | Size | Data type | Index formula |
| :----- | :----------------------------------------------- | :--: | :-------: | :-----: |
| Voltage dynamic spectrum | Polarisation <br> (Frequency) channel <br> (Time) sample | `Np`<br>`Nc`<br>`Ns` | `cuFloatComplex` | `p*Nc*Ns + c*Ns + s` |
| Power dynamic spectrum | (Frequency) channel <br> (Time) sample | `Nc`<br>`Ns` | `float` | `c*Ns + s` |
| Image | Width <br> Height | `W`<br> `H` | `float` | `y*W + x` |

## Binning {#binning}

Binning refers to the amount of time and/or frequency averaging that happens after the Stokes parameter(s) have been formed.
The amount of binning is set jointly by

1. The user, and
2. The maximum size of OpenGL textures and CUDA surfaces.

To satisfy the second constraint, we fix the maximum size of the texture and surface to be 4096x4096 pixels.
If, upon loading a new data set, there are fewer than 4096 channels, or fewer than 4096 time samples, then the texture and surface are created with the smaller dimensions, accordingly.
