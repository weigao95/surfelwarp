### The image CRF implementation using window search

The window-search based Conditional Random Field implementation for images. Currently, the implementation is customized for binary label (foreground and background), and it is expected to provide an conservative over-estimated foreground.

For efficiency, the CRF works on **2x2 subsampled** RGB (and depth) images. 