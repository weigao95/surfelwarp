### The GeometryUpdater in processing pipelines
In the illustration of parallel processing, the GeometryUpdater's diagram is

**GeometryUpdater**: (geometry\_(0, 1), warpfield\_1, observation\_1) -> geometry\_(1, 1)

Which means, the surfel model's live surfels should match the warped surfels from reference surfels and the provided warp field, also matched the provided observation. 

The implementation use a **double buffer** scheme, where there are two actual filling of geometry model. The geometry updater take one as input, produce an array of indicator of validity and a compacted array of append surfels and their knn. However, the updater will **modify its input's LIVE surfels**, which is not used in the solver.

For the detailed implementation, please refer to the document under geometry folder.