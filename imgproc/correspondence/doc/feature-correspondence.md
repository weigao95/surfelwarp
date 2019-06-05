### The Sparse Feature Correspondence

The feature correspondence in this package takes input as clipped, normalized rgb or depth images in the original resolution, Output an array of pixels pairs that are in correspondence. The **ImagePairCorrespondence.h** is the interface file.

The current implementation is the Global Patch Collider by Wang et al. The pretained model is provided by opencv_contrib. 
The detailed algorithm is, 
1. Forall patch: extract features, search within the forest (typically 4-8 trees), encode the search result into a hash value.
2. Sort all the hash value, find consecutive patch pairs with the same hash value but from different images. Note that if more than 3 patches are mapped to the same hash value, then it is directly rejected.
3. Filter the correspondence according to forground mask.