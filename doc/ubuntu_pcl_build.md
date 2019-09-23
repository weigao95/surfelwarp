### Building SurfelWarp on Ubuntu 16.04 with `PCL 1.8`

You need a nvidia GPU with `CUDA 9`  installed. Note that `CUDA 10` is not compatible with `PCL 1.8` and doesn' work. If you want to build with `CUDA 10`, please follow [this document](https://github.com/weigao95/surfelwarp/blob/master/doc/ubuntu_cliantro_build.md).

To choose the CUDA architecture [compatible](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) with your `CUDA` version and graphics card, thereby also reducing compile time, pass `-DCUDA_ARCH=<arg>` to CMake, where `<arg>` is a two-digit compile compatibility version, e.g. "61", or choose this number from the corresponding drop-down in `cmake-gui`.

 In addition to `CUDA` , the code depends on `PCL`, `OpenCV` and `GLFW`. On Ubuntu, you can run the following command to install these dependencies:

```shell
sudo apt-get install libpcl-dev libopencv-dev libglfw3 libglfw3-dev
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

