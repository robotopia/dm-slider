# Data pipeline

\dotfile cudaopengl.dot

| Process | Format | Associated CUDA function | Size (1 second,<br>24 x 1.28 MHz channels) |
| :------ | :----- | :---------- | :------------ |
| **START** | VDIF =<br>complex unsigned char,<br>dual polarisation,<br>in "frames" | cudaMemcpy() | 125 MB (GPU)<br>(3 seconds, 24 coarse channels) |
| Strip frame headers &<br>Promote to cuFloatComplex | cuFloatComplex,<br>dual polarisation | `cudaVDIFToFloatComplex()` | 500 MB |
| Fourier transform | cuFloatComplex,<br>dual polarisation | `cuFFT` | 500 MB |
| Coherently dedisperse<br>each channel | cuFloatComplex,<br>dual polarisation |  | 500 MB |
| Remove interchannel<br>dispersion delays | cuFloatComplex,<br>dual polarisation |  | 500 MB |
| Inverse Fourier transform | cuFloatComplex,<br>dual polarisation | `cuFFT` | 500 MB |
| Detection (I,Q,U,V) | float | cudaStokesI() | 250 MB |
| Binning | float |  |  |
| Copy to CUDA Surface | float | cudaCopyToSurface() |  |
| Map to OpenGL Texture | float | CUDA-OpenGL interoperability |  |
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
