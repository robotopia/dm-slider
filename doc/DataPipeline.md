# Data pipeline

\dotfile cudaopengl.dot

| Process | Array format | Associated CUDA function | Size (1 second,<br>24 x 1.28 MHz channels) |
| :------ | :----- | :---------- | :------------ |
| **START** | VDIF =<br>complex unsigned char,<br>dual polarisation,<br>in "frames" | cudaMemcpy() | 125 MB (GPU)<br>(3 seconds, 24 coarse channels) |
| Strip frame headers &<br>Promote to cuFloatComplex | [Voltage dynamic spectrum](@ref arrayformats) | `cudaVDIFToFloatComplex()` | 500 MB |
| Fourier transform | [Voltage dynamic spectrum](@ref arrayformats) | `cuFFT` | 500 MB |
| Coherently dedisperse<br>each channel | [Voltage dynamic spectrum](@ref arrayformats) |  | 500 MB |
| Remove interchannel<br>dispersion delays | [Voltage dynamic spectrum](@ref arrayformats) |  | 500 MB |
| Inverse Fourier transform | [Voltage dynamic spectrum](@ref arrayformats) | `cuFFT` | 500 MB |
| Detection (I,Q,U,V) | [Power dynamic spectrum](@ref arrayformats) | cudaStokesI()<br>cudaStokesQ()<br>cudaStokesU()<br>cudaStokesV() | 125 MB |
| Binning | [Power dynamic spectrum](@ref arrayformats) |  |  |
| Copy to CUDA Surface | [Image](@ref arrayformats) | cudaCopyToSurface() |  |
| Map to OpenGL Texture | [Image](@ref arrayformats) | CUDA-OpenGL interoperability |  |
| Draw on quad (via shaders) |  |  |  |

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

| Format | Dimensions (from slowest to<br>fastest changing) | Size | Data type | Formula |
| :----- | :----------------------------------------------- | :--: | :-------: | :-----: |
| Voltage dynamic spectrum | Polarisation <br> (Frequency) channel <br> (Time) sample | `Np`<br>`Nc`<br>`Ns` | `cuFloatComplex` | `p*Nc*Ns + c*Ns + s` |
| Power dynamic spectrum | (Frequency) channel <br> (Time) sample | `Nc`<br>`Ns` | `float` | `c*Ns + s` |
| Image | Width <br> Height | `W`<br> `H` | `float` | `y*W + x` |

