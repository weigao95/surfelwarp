### The SurfelModel struct
The surfel model class(or struct) is a light weight collection of Buffer and Array pairs on the surfels, KNN and weights. The buffer is from OpenGL in production, but can be allocated inside for debugging. There are two surfel models in use for compaction input/output of each other, please refer to Parallel for detailed description.

The surfel model only implements the buffer manage and sync with OpenGL context, for complex operations like model fusion or reset are not in the scope of this class.

The struct maintains to time flags, **geometry_index** is the latest depth observations that this model has been fused with, **warp_index** is the index of warp field between the reference and live frame surfels. In serial implementation, these two are the same; while in parallel cases, **warp_index** may be larger.