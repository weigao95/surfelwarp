## The parallel processing in SurfelWarp

There are three major components in surfelwarp, namely the **ImageProcessor**, the **WarpSolver** and **GeometryUpdater**. They are expected to execute in parallel. The pipeline for iterations might be: 

*Iteration 0:*
**ImageProcessor**: (No Input) -> observation\_3
OpenGLSynchronize(WarpSolver, GeometryUpdater)
Rendering maps using geometry\_(0, 1)(do not explicitly need warpfield\_1 as it is inside geometry\_(0, 1))
OpenGLSynchronize(WarpSolver, GeometryUpdater)
**WarpSolver**: (geometry\_(0, 1), warpfield\_1, observation_2) -> warpfield\_2
**GeometryUpdater**: (geometry\_(0, 1), warpfield\_1, observation\_1) -> geometry\_(1, 1)
BarrierSynchronize(All Modules)
Compact the geometry output to another buffer
(prallel with above)Update the nodes of warpfield\_2 using appended surfels in geometry\_(1, 1)
Update the skinning of reference surfels in geometry\_(1, 1) using new nodes in warpfield\_2
Perform a forward warp using warpfield\_2 to update from geometry\_(1, 1) to geometry\_(1, 2)

*Iteration 1:*
**ImageProcessor**: (No Input) -> observation\_4
**WarpSolver**: (geometry\_(1, 2), warpfield\_2, observation_3) -> warpfield\_3
**GeometryUpdater**: (geometry\_(1, 2), warpfield\_2, observation\_2) -> geometry\_(2, 2)
BarrierSynchronize(All Modules)
Update the nodes of warpfield\_3 using appended surfels in geometry\_(2, 2)
Update the skinning of geometry\_(2, 2) using new nodes in warpfield\_3
Perform a forward warp geometry\_(2, 2) -> geometry\_(2, 3)
---

The first index is the geometry\_index, where the second index is the warp\_index. Refer to the SurfelMode for details.

The re-initialization will happen after the barrier of some iterations. Suppose we would like to re-initialize the model at iteration 1, the operations would be

*Re-init for Iteration 1:*
**ImageProcessor**: (No Input) -> observation\_4
**WarpSolver**: (geometry\_1, warpfield\_2, observation_3) -> warpfield\_3
**GeometryUpdater**: (geometry\_1, warpfield\_2, observation\_2) -> geometry\_2
BarrierSynchronize(All Modules)
**Re-init Processing**
1. Set the reference and live surfels of geometry\_1 to be the live frame surfels of geometry\_2 (of course with error correction, now the new geometry\_1 should contains old warpfield\_2, thus set warpfield\_2 to indentity is safe). 
2. Reset warpfield\_2 using the live frame of geometry\_1. (Here we essentially discard the warpfield\_3 and observation\_4)
---
*Iteration 2 after reinit-1:*
**ImageProcessor**: (No Input) -> observation\_4
**WarpSolver**: (geometry\_1, warpfield\_2, observation_3) -> warpfield\_3
**GeometryUpdater**: (geometry\_1, warpfield\_2, observation\_2) -> geometry\_2
BarrierSynchronize(All Modules)


This is a high level illustration, please refer to detailed documentations for data structures and io for each modules.
