### The image CRF implementation using Permutohedral Grid

The code implement Philipp Krähenbühl at al. "Efficient Inference in Fully Connected CRFs with
Gaussian Edge Potentials" and Andrew Adams et al. "Fast High-Dimensional Filtering Using the
Permutohedral Lattice", for foreground segmentation, which is a binary label CRF inference problem. 



For efficiency, the CRF works on **2x2 subsampled** segmentation mask (while it takes normalized, registeres 
rgb and depth pairs as input). In this project, the original size is cliped_rows and clip_cols. 