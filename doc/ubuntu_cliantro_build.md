### Building SurfelWarp on Ubuntu 16.04/18.04 with `cliantro`

You need a nvidia GPU with `CUDA >= 9`  installed (`CUDA 10` is OK). To choose the CUDA architecture [compatible](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) with your `CUDA` version and graphics card, thereby also reducing compile time, pass `-DCUDA_ARCH=<arg>` to CMake, where `<arg>` is a two-digit compile compatibility version, e.g. "61", or choose this number from the corresponding drop-down in `cmake-gui`.

Then, you need to install `cliantro` by following [this document](https://github.com/weigao95/surfelwarp/blob/master/doc/cilantro_build.md). To switch to `cilantro` and effectively remove the `PCL` dependency, pass `-DVISUALIZATION_LIBRARY=cilantro` and `-Dcilantro_DIR=<path_to_cilantro_install_directory>` when you run `cmake` or fill in the corresponding `cmake-gui` options.

The code also depends on `OpenCV` and `GLFW`. On Ubuntu, you can run the following command to install these dependencies:

```shell
sudo apt-get install libopencv-dev libglfw3 libglfw3-dev
```

Now you are ready to build (remember to add cmake arguments as necessary):

```shell
git clone https://github.com/weigao95/surfelwarp
cd surfelwarp
mkdir build && cd build
cmake ..
make
```

If the build is successful, you might continue with the [run instruction](https://github.com/weigao95/surfelwarp).

