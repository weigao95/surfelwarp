## The Implementation of Matrix-Free Warp Solver

The matrix free warp solver takes a list of jacobian terms, and compute y = JtJ x using these terms.

The solver might require a inverse index, i.e., a map from the nodes to the terms that this node is involved. The method to build this index is rather straight forward. For the method to compute Ax, using to following algorithms:

parallel for each output node i twist (6x1 vector):
-forall least square term t that this node is involved:
--read d_term_d_twist_i (a 6x1 vector)
---for j in nodes that involved with this term
----read d_term_d_twist_j (another 6x1 vector), and x_j
----Compute it


The inserse index will also be used to compute the diagonal entries of the JtJ for pre-conditioning.
