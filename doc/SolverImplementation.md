## The Implementation of WarpSolver

The warp solver takes input from geometry(rendered), observation, and warp field, solve for the optimal pose of warp field. Optionally, perform a rigid presolve.

There are two types of solver, one is materialized solver, another one is matrix-free. They still share lots of components. Please refer to sub-doc for their details.

