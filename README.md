# [SurfelWarp](<https://sites.google.com/view/surfelwarp/home>)

SurfelWarp is a dynamic reconstruction system similar to [DynamicFusion](https://www.youtube.com/watch?v=i1eZekcc_lM). However, surfelwarp uses flat [surfel](https://en.wikipedia.org/wiki/Surfel) array (instead of volumetric field) as the geometry representation, which makes the pipeline more robust and efficient. The approach is described in [our paper](https://arxiv.org/abs/1904.13073).

### Demo [[Video]](https://drive.google.com/open?id=1Qs-yM8RbkG4eJoMIs5y_WA_J1KMBLYCW)[[Presentation]](https://www.youtube.com/watch?v=fexYm61VGMA)

[![Surfelwarp](./doc/surfelwarp.png)](https://www.youtube.com/watch?v=fexYm61VGMA)

### Publication

Wei Gao and Russ Tedrake, "SurfelWarp: Efficient Non-Volumetic Single View Dynamic Reconstruction", Robotics: Science and Systems (RSS) 2018  [[Project]](<https://sites.google.com/view/surfelwarp/home>) [[Paper]](https://arxiv.org/abs/1904.13073)[[Presentation]](https://www.youtube.com/watch?v=fexYm61VGMA)

### Build Instruction

The code is developed on Visual Studio 2015 with `CUDA-9` . It also works on Ubuntu 16.04, but for unknown reason it runs much slower (I haven't investigate it carefully). In addition to `cuda`, the code depends on `pcl` , `opencv` and `glfw`. On Ubuntu, these dependencies can be installed with `apt`, while they need to be built from source on windows.

The tested compilers are `gcc-5.4`, `gcc-6.0` and `gcc-6.5`. Note that `gcc-5.5` is not supported by CUDA and may cause errors. For the installation of CUDA, please follow the [official guide](<https://developer.nvidia.com/cuda-downloads>).

This repo depends on `pcl` and `opencv`. The default versions of both Ubuntu 16.04 and `ros-kinetic` have been tested. For Ubuntu 16.04, you can run the following command to install them

```shell
sudo apt-get install libpcl-dev libopencv-dev
```

This repo also depends on `glog` which can be installed by

```shell
sudo apt-get install libgoogle-glog-dev
```

Now you are ready to build

```shell
git clone https://gaowei19951004@bitbucket.org/gaowei19951004/poser-public.git
cd ${project_root}
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

The [`apps/rigid_pt2pt`](<https://sites.google.com/view/filterreg/home>) would be a good starting point. The test data is also included in that subdirectory. Other executables in `apps` subfolder have a very similiar code structure.

### FAQ

- What's the typical speed of surfelwarp?

  On the windows platform with a Nvidia Titan Xp GPU, the processing time is usually less than 20 [ms]. To test the speed in [surfelwarp_app](https://github.com/weigao95/surfelwarp/blob/master/apps/surfelwarp_app/main.cpp), you need to build with Release and [disable offline rendering](https://github.com/weigao95/surfelwarp/blob/05f5bb9209a6d8f38febad63a92639054877bb54/apps/surfelwarp_app/main.cpp#L33) (which takes most of the time).

- How to use my own data?

  You might override the [FetchInterface](https://github.com/weigao95/surfelwarp/blob/master/imgproc/FetchInterface.h) and use it with [ImageProcessor](https://github.com/weigao95/surfelwarp/blob/master/imgproc/ImageProcessor.h). If you don't have performance requirement, you can also convert your data into the format of [VolumeDeform](https://www.lgdv.tf.fau.de/publicationen/volumedeform-real-time-volumetric-non-rigid-reconstruction/).

- How to deal with topology change?

  Currently, only periodic reinitialization is implemented. To use it, set the [flag](https://github.com/weigao95/surfelwarp/blob/bfb2ccaac5b986fb42888f41624a275c1594e084/test_data/boxing_config.json#L11) in config file. More advanced criterion of reinitialization would come soon.

### TODO

The code is re-factored and improved from the repo of our RSS paper. There are some new features and some old code  need to be ported into this repository. Here is a list of TODOs:

- [x] Add sparse feature correspondence
- [x] Implement albedo reconstruction
- [ ] Rewrite automatic reinitialization detection for the new geometry and loss function. Currently, only periodic reinitialization is implemented
- [ ] Port the GPU kdtree querying (although the overall speed is already relatively fast)

### Contact

If you have any question or suggestion regarding this work, please send an email to weigao@mit.edu