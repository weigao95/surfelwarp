### Building SurfelWarp on Windows with `cilantro`

Building with `cilantro` has been tested on Visual Studio 2017 also with `CUDA 9.2`, but, should in theory, also work with any version of `CUDA 10` as well. `CUDA` and `cmake` can be installed from their official websites. The community version of Visual Studio is sufficient. `OpenCV` can be downloaded from the official website as a prebuilt-binaries-and-header package and extracted to a location of your choosing. 

**This code and all its dependencies MUST be built in 64 bit**. In other words, you should select the **Win64** (or, for later CMake versions, **x64**) generator in `cmake`.

To build `cilantro` you also need to build its dependence `Pangolin`. The build-and-installation procedures for `Pangolin`, `cilantro`, and `GLFW` are very similar. Note that `cilantro` requires the `Eigen` to be properly installed via CMake, and you might as well use the aready-provided version of `Eigen` in `external/eigen3.4` for this. All of these packages are managed by `CMake` and you can follow the standard procedure to generate the Visual Studio solution. After solution generation, open it in Visual Studio running as an administrator (right-click the VS icon and choose `Run as administrator`). You can then build and install the package using the `INSALL` project inside the solution. 

Now you are ready to build this repo. To tell CMake where to find the dependencies, you can set `xxx_DIR` to the location of _xxxConfig.cmake_ on disk, where xxx can be `glfw3_DIR`, `OpenCV_DIR`, and/or any other dependencies. Alternatively, add these as permanent [environment variables](https://www.architectryan.com/2018/08/31/how-to-change-environment-variables-on-windows-10/) to simplify future builds.  After that, just follow the standard `CMake` and Visual Studio workflow to build this repo. 

 