# Change log

- Reorganised the internal (GPU) data format to make polarisation the slowst changing quantity.
- Implemented coherent dedispersion! (Use right mouse button click and drag to change dispersion measure.)

## v0.4

Implemented basic VDIF support (Issue #5).
You can load multiple VDIF files, and it will display them on the quad.
Although the dynamic range is now controlled (in the correct way) via uniforms in the fragment shader, there is still only basic control via the right button mouse click, to shift the dynamic range up and down.

## v0.3

- Switched from GLFW to GTK (Issue #6).
- Reorganised code and filenames (changed from `coherent_dedispersion.cu` to `dm-slider.c` and `cohdd.cu`).

## v0.2

`coherent_dedispersion.cu` now implements OpenGL-CUDA interoperability using "surfaces" to access and manipulate image data with custom kernels.
It draws a quad with a heat map gradient.
The quad can be rotated with a left mouse click `n` drag (as in `v1.0`), while the values themselves can be raised or lowered via a right mouse click `n` drag.

## v0.1

This version sees the first successful implementation of OpenGL-CUDA interoperability.
The relevant program is `coherent_dedispersion.cu`.
It draws a magenta quad, which can be rotated by clicking and dragging with the mouse.
The rotation is done via a custom CUDA kernel, acting on the points of the quad defined via OpenGL.
