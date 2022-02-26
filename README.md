# Slide buffer

This is an implementation of a ring buffer in CUDA, with bi-directional "sliding" of the data in mind, rather than just uni-directional "stream" processing of a data set.
That is, one can step *back* through a large data set as easily as stepping forward.


