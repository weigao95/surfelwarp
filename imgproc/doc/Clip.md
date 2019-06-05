### The Clipping in Image-Processing

For better window-search, in this implementation the input depth and color images are clipped on each boundary. In this fllowing text, the sizes are refered as **raw_size** and **clip_size**.

In image process package (imgproc), the main class ImageProcessor read raw depth/rgb images in **raw_size**, produce filtered depth images, normalized color images, and all other maps in **clip_size**.

The renderer draws images in **clip_size**.

Note there is an transform of **intrinsic** matrix during cliping. 