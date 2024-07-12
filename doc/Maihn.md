## Non-Rigid Reconstruction and PCD Retrieval for Each Frame with Parallel Model Fusion

The `apps/surfelwarp_thread` demonstrates parallel processing for non-rigid reconstruction and point cloud data (PCD) retrieval.

## Added RealSense Fetch Frame Mode

Please refer to `realsense_config` and `imgproc/frameio/RealsenseFetch.h` and `RealsenseFetch.cpp` for implementation details.

## Added Mode to Disable Segmentation

Refer to `realsense_config` and the "use_segmentation" option. If foreground/background segmentation is not used, set "use_downsample" to true. Note that with large datasets, CUDA may encounter issues if this is not done.

## Added User-Selectable Online Rendering Mode

Refer to `realsense_config`. When "offline_rendering" is set to true, "show_online" should be set to false. Online rendering takes priority over offline rendering. Therefore, if "show_online" is true, the system will render online regardless of the "offline_rendering" setting.

## Added Mode to Downsample Input Frames

Due to the limitations of CUDA global constant memory and limited CUDA programming knowledge, a downsample mode has been added. For example, if the RealSense input frame is 640x480, downsampling will reduce it to 320x240. This mode is implemented for `"imgproc/frameio/RealsenseFetch.h"` and `"imgproc/frameio/VolumeDeformFileFetch.cpp"`.

