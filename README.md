# [SurfelWarp](<https://sites.google.com/view/surfelwarp/home>)

SurfelWarp is a dynamic reconstruction system similar to [DynamicFusion](https://www.youtube.com/watch?v=i1eZekcc_lM). Compared to other dynamic reconstruction methods, surfelwarp uses flat [surfel](https://en.wikipedia.org/wiki/Surfel) array (instead of volumetric field) as the geometry representation, which makes the pipeline more robust and efficient. The approach is described in [our paper](https://arxiv.org/abs/1904.13073).

### Demo [[Video]](https://drive.google.com/open?id=1Qs-yM8RbkG4eJoMIs5y_WA_J1KMBLYCW)[[Presentation]](https://www.youtube.com/watch?v=fexYm61VGMA)

[![Surfelwarp](./doc/surfelwarp.png)](https://www.youtube.com/watch?v=fexYm61VGMA)

### Publication

Wei Gao and Russ Tedrake, "SurfelWarp: Efficient Non-Volumetic Single View Dynamic Reconstruction", Robotics: Science and Systems (RSS) 2018  [[Project]](<https://sites.google.com/view/surfelwarp/home>)[[Paper]](https://arxiv.org/abs/1904.13073)[[Presentation]](https://www.youtube.com/watch?v=fexYm61VGMA)

### Build Instruction

The code is developed on Visual Studio 2015 with `CUDA 9` . Note that `CUDA 10` and higher versions are not compatible with `pcl 1.8` and don't work. The code also works with Ubuntu 16.04, but for unknown reason it runs much slower on ubuntu (haven't investigate yet). 

In addition to `cuda`, the code depends on `pcl` , `opencv` and `glfw`. On Ubuntu, these dependencies can be installed with `apt`, while they need to be built from source on windows. For Ubuntu 16.04, you can run the following command to install the dependency

```shell
sudo apt-get install libpcl-dev libopencv-dev libglfw3 libglfw3-dev
```

Now you are ready to build

```shell
git clone https://github.com/weigao95/surfelwarp
cd surfelwarp
mkdir build && cd build
cmake ..
make
```

For build on windows please refer to [this document](https://github.com/weigao95/surfelwarp/blob/master/doc/windows%20build.md). We also provide a [pre-built binary](https://github.com/weigao95/surfelwarp/tree/master/test_data/binary) for the windows platform.

### Run Instruction

We use the [VolumeDeform dataset](https://www.lgdv.tf.fau.de/publicationen/volumedeform-real-time-volumetric-non-rigid-reconstruction/) to illustrate how to run the code. An example configuration file is provided [here](https://github.com/weigao95/surfelwarp/blob/master/test_data/boxing_config.json) for the "boxing" sequence. First, you need to download the boxing sequence from the VolumeDeform dataset and extract it to `data_root`, your file structure should look like

```
${data_root}/frame-000000.color.png
${data_root}/frame-000000.depth.png
...
```

You also need to download the trained model for Global Patch Collider (gpc) from [here](https://drive.google.com/file/d/10A80gH5p4_eDbYPs10wHLI-ZKBwkG1fC/view?usp=sharing). Let the path to the model be `${gpc_path}` .

In the [configuration file](https://github.com/weigao95/surfelwarp/blob/master/test_data/boxing_config.json), please modify the `data_prefix` and `gpc_model_path` to `${data_root}` and `${gpc_path}`, respectively. After that, you can run the algorithm with

```
cd ${project_root}/build/apps/surfelwarp_app
./surfelwarp_app /path/to/config
```

If everything goes well, the executable would produce the reconstructed result per frame in the same folder as `surfelwarp_app`. 

### FAQ

- What's the typical speed of surfelwarp?

  On the windows platform with a Nvidia Titan Xp GPU, the processing time is usually less than 20 [ms] per frame. To test the speed in [surfelwarp_app](https://github.com/weigao95/surfelwarp/blob/master/apps/surfelwarp_app/main.cpp), you need to build with Release and [disable offline rendering](https://github.com/weigao95/surfelwarp/blob/05f5bb9209a6d8f38febad63a92639054877bb54/apps/surfelwarp_app/main.cpp#L33) (which takes most of the time).

- How to use my own data?

  You might override the [FetchInterface](https://github.com/weigao95/surfelwarp/blob/master/imgproc/FetchInterface.h) and use it with [ImageProcessor](https://github.com/weigao95/surfelwarp/blob/master/imgproc/ImageProcessor.h). If you don't have performance requirement, you can also convert your data into the format of [VolumeDeform](https://www.lgdv.tf.fau.de/publicationen/volumedeform-real-time-volumetric-non-rigid-reconstruction/).

- How to deal with topology change?

  Currently, only periodic reinitialization is implemented. To use it, set [this flag](https://github.com/weigao95/surfelwarp/blob/bfb2ccaac5b986fb42888f41624a275c1594e084/test_data/boxing_config.json#L11) in config file. More advanced criterion of reinitialization would come soon.

### TODO

The code is re-factored and improved from the repo of our RSS paper. There are some planned new features and some old code  need to be ported into this repository. Here is a list of TODOs:

- [x] Add sparse feature correspondence
- [x] Implement albedo reconstruction
- [ ] Rewrite automatic reinitialization detection for the new geometry and loss function. Currently, only periodic reinitialization is implemented
- [ ] Port the GPU kdtree querying (although the overall speed is already relatively fast)

### Contact

If you have any question or suggestion regarding this work, please send an email to weigao@mit.edu
