# Change log

## v0.1

This version sees the first successful implementation of OpenGL-CUDA interoperability.
The relevant program is `coherent_dedispersion.cu`.
It draws a magenta quad, which can be rotated by clicking and dragging with the mouse.
The rotation is done via a custom CUDA kernel, acting on the points of the quad defined via OpenGL.
