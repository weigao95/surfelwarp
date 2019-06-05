### The WarpField class
The warp field maintains an array of nodes in reference frame and live frame, their SE(3) transformation, the node graph neighbours (exclude itself), KNN and weights (include itself). Also, appropriate search structures like oct-tree/kd-tree is maintained in this class if used.

Some members of the warp field are synced array between device and host. However, they are not always updated and required explicit sync.

The warp field provide skinning and warpping interface for outside access, with a flag variable indicator whether perform update to the node in itself. 

More complex operations like model fusion will take warp field and surfel model as input, produce another surfel model and warp field as output. The warp solver also this it as input, but will modifiy the SE(3) deformation.