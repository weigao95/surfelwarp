### The Image Processing Package

The image processing package takes input from disk images or openni2 interface, provides the later fusion module with the following output:

1. The filtered depth image
2. The vertex, normal, confidence and radius map for each pixel. The vertex and confidence are in a float4 texture, while normal and radius are in another float4 texture.
3. The color, last accessed and initialization time map, in a float4 texture. The color is encoded in a float value from a uchar3. 
4. A compacted array of valid surfels.
4. The density map (float1 texture) and its gradient w.r.t x and y (float2 texture)
5. The normalized rgb image for current and previous frame (float4 texture)
6. The foreground mask (uchar1 texture), filtered foreground mask (float1 texture) and the gradient of the filter mask (float2 texture).
7. An array of corresponded pixel pairs.

The dependency is 

The image processor is expected to run in parallel with the main module, and parallize the inside computation using streams.