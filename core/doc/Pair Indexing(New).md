## The Pair Index used in new solver

In addition to the node2pixel index, the NodePair2Pixel and NodePair2NonImagedIndex is also required. This document describe the index building scheme.

After experiment, we found that build the index using non-image term is not complete and stable. Thus, we still need to build the full index.

The full index are combined of image term and non-image term. However, compare with previous approach, the image term index are shared for dense depth and density map. Which means, the nodepair knows the pixel that is involved with, also the non image terms.