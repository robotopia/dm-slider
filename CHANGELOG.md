# Change log

## v0.3

Switched from GLFW to GTK (Issue #6).
Reorganised code and filenames (changed from `coherent_dedispersion.cu` to `dm-slider.c` and `cohdd.cu`).

## v0.2

`coherent_dedispersion.cu` now implements OpenGL-CUDA interoperability using "surfaces" to access and manipulate image data with custom kernels.
It draws a quad with a heat map gradient.
The quad can be rotated with a left mouse click `n` drag (as in `v1.0`), while the values themselves can be raised or lowered via a right mouse click `n` drag.

## v0.1

This version sees the first successful implementation of OpenGL-CUDA interoperability.
The relevant program is `coherent_dedispersion.cu`.
It draws a magenta quad, which can be rotated by clicking and dragging with the mouse.
The rotation is done via a custom CUDA kernel, acting on the points of the quad defined via OpenGL.
