### Build SurfelWarp on Windows

The building has been tested on Visual Studio 2015 with `CUDA 9.2`  and `cmake 3.9.6`. The `CUDA 10` and higher versions are not compatible with `pcl`, thus please use `CUDA 9`. The Visual Studio, `CUDA` and `cmake` can be installed from their official websites. The community version of Visual Studio is sufficient.

**This code and all its dependencies MUST be built in 64 bit**. In other words, you should select the **Visual Studio 14 2015 Win64** generator in `cmake`. 

The building and installing procedure for `pcl`, `opencv` and `glfw` are very similar. These packages are managed by `cmake` and you can follow the standard procedure to generate the Visual Studio Solution. After solution generation, you can build and install the package using the `INSALL` project inside the solution. The standard configuration of these packages should be sufficient, although some debug code might require the gpu support of `pcl`.

Now you are ready to build this repo. You need to change [these lines](https://github.com/weigao95/surfelwarp/blob/6265b3aed872f61f8e504f0216631e36579e191d/CMakeLists.txt#L16) of `CMakeLists.txt`  to your local install directories of `pcl`, `opencv` and `glfw`. After that, just follow the standard procedure of `cmake` and Visual Studio to build this repo. 