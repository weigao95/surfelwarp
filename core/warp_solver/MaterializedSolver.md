## The Implementation of Materialized Warp Solver

The materialized solver is built upon the inverse index from nodes to term that this node is involved. 

The materialized solver also requires a map from node PAIRs to terms that BOTH nodes is involved. The building of this index and the whole matrix can be expensive in this sense.

To help the parallel implementation, first build a CPU implementation useing Eigen. The Eigen implementation directly construct the jacobian matrix, then it construct JtJ by matrix multiple (After all the CPU implementation does not need to care performance).

The matrix in flatten blocks are all COLUMN-MAJOR, i.e., the continuous elements are in the same column but different rows.
