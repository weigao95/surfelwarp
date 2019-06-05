## The implementation of surfel fusion

In most case, the geometry is fused with new observation from the depth/rgb images. This document highlight the implementation of GeometryUpdater class (and subclass it refers).

GeometryUpdater's input consists of read-only access to the warp field, image observation, and read-write access to both input/output geometry in double buffer schemes. At a finer grain, the updater will only write the live geometry of the input buffer, and perform a compaction to the output buffer. Later operation like inverse warping and warp-field update, will be called in GeometryUpdater, but implemented in WarpField.

The fusion contains the following steps:
1. For each pixel, fuse it into some model surfel, or mark it will potentially be appended (candidate)
2. For each model surfel, determine whether it will remain based on its neighbour on the index map
3. For each candidate observation surfel, check its skinning and collision
4. Compact all the remaining model surfels and valid candidate surfels into another buffer.

After the processing of surfel fusion, the live vertex/normal, knn/weights, color, various time stamp in the output buffer, should be ready for inverse warping.