### Building SurfelWarp on Windows with `PCL 1.8`

Building surfelwarp with `PCL` has been tested on Visual Studio 2015 with `CUDA 9.2`  and `cmake 3.9.6`. Note that `CUDA 10` and higher versions are not compatible with `PCL` and won't work.  `CUDA` and `cmake` can be installed from their official websites. The community version of Visual Studio is sufficient. `OpenCV` can be downloaded from the official website as a prebuilt-binaries-and-header package and extracted to a location of your choosing. 

**This code and all its dependencies MUST be built in 64 bit**. In other words, you should select the **Win64** (or, for later CMake versions, **x64**) generator in `cmake`.

The build-and-installation procedures for `PCL`, and `GLFW` are very similar. All of these packages are managed by `CMake` and you can follow the standard procedure to generate the Visual Studio solution. After solution generation, open it in Visual Studio running as an administrator (right-click the VS icon and choose `Run as administrator`). You can then build and install the package using the `INSALL` project inside the solution. The standard configuration of these packages should be sufficient, although some debug code might require the GPU support of `PCL`.

Now you are ready to build this repo. To tell CMake where to find the dependencies, you can set `PCL_DIR` to the location of _PCLConfig.cmake_ on disk, and likewise for `glfw3_DIR`, `OpenCV_DIR`, and/or any other dependencies. Alternatively, add these as permanent [environment variables](https://www.architectryan.com/2018/08/31/how-to-change-environment-variables-on-windows-10/) to simplify future builds.  After that, just follow the standard `CMake` and Visual Studio workflow to build this repo. 

 
