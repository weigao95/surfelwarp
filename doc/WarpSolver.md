## The WarpSolver

In the high level illustration, the task of warp solver is
**WarpSolver**: (geometry\_(0, 1), warpfield\_1, observation_2) -> warpfield\_2

In the implementation, the warp solver uses a slightly more general form:
**WarpSolver**: (geometry, warpfield, observation) -> warpfield
in which apply the warpfield to the reference frame surfels of geometry should be a good initialization of observation (or in other words, has temporal coherence); while rendering the live surfels of geometry should be a good visibility prediction(This is not difficult). This form make it easier to work with both parallel and serial implementations.

The warp solver maintains its own copy of SE(3) deformation and should not modify any inputs. Which implies, the warp solver should never applied any warp field to geometry model, or write its SE(3) to warp field before explicit synchronize.

