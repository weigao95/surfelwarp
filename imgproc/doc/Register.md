### Align the depth image with the RGB image using calibrated intrinsics and extrinsics of both cameras

The alignment in this code is to reproject the depth image to the rgb image. More specifically, the depth image is first back-projected into 3d points using the intrinsic of depth camera, then transformed to the rgb camera using the pre-calibrated depth2rgb se3 transformation, finally project to the image plane using the intrinsic of rgb camera.

**After reprojection, ONLY RGB intrinsic will be used in later operations, including VertexMap Contruction, Rendering and Project Vertex into Images.**

This reprojection may causes neighbouring depth pixels projecting to the same RGB pixel. To avoid this, we first up-sampling the projected image plane (an 2 x 2 up-sampling is used here), then collect the pixel using corresponding windows search. Assuming the view angle between depth camera and rgb camera is less than 60 degree, the 2x2 up-sampling should solve the problem.

The idea under these (looks) complex operations is that, the RGB images are usually complete and of high quality (compared with depth images). Thus, this projection should be better than colored point-cloud when performing segmentation and correspondence search.
