## The index used in new solver

Currently the indexing scheme is somehow redundant. The dense depth term and dense image term use almost the same index, but they are built separately.

In new indexing, the density and dense depth index are combined into dense image term. The task to zero-out invalid pixels goes into DenseDepthHandler and DensityForegroundMaskHandler. 

For implementation, first the density term index are set to be empty in node2term and nodepair2term index builder. By this, the solver should not use density term. After elimination and testing, term_offset_types should be updated correspondingly. Finally, the density term is merged into densedepth term.
 