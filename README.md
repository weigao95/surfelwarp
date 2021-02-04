# [SurfelWarp](<https://sites.google.com/view/surfelwarp/home>)

SurfelWarp is a dynamic reconstruction pipeline. Compared to other dynamic reconstruction methods like [DynamicFusion](https://www.youtube.com/watch?v=i1eZekcc_lM), surfelwarp uses flat [surfel](https://en.wikipedia.org/wiki/Surfel) array (instead of volumetric field) as the geometry representation, which makes the pipeline robust and efficient. The approach is described in [our paper](https://arxiv.org/abs/1904.13073).

### Demo [[Video]](https://drive.google.com/open?id=1Qs-yM8RbkG4eJoMIs5y_WA_J1KMBLYCW)[[Presentation]](https://www.youtube.com/watch?v=fexYm61VGMA)

[![Surfelwarp](./doc/surfelwarp.png)](https://www.youtube.com/watch?v=fexYm61VGMA)

### Publication

Wei Gao and Russ Tedrake, "SurfelWarp: Efficient Non-Volumetic Single View Dynamic Reconstruction", Robotics: Science and Systems (RSS) 2018  [[Project]](<https://sites.google.com/view/surfelwarp/home>)[[Paper]](https://arxiv.org/abs/1904.13073)[[Presentation]](https://www.youtube.com/watch?v=fexYm61VGMA)

### Build Instructions

The code was originally developed with `CUDA 9` and `PCL 1.8` on Visual Studio 2015 and Ubuntu 16.04. Thanks to the contribution by [@Algomorph](https://github.com/Algomorph), the code works with higher version of `CUDA`, Ubuntu 18.04 and Visual Studio 2017. Also note that, for some unknown reason, the code runs much slower on Ubuntu (seems to be problem with GPU driver that only permits Debug mode).

According to your environment, please follow the specific build instruction:

- [Ubuntu 16.04 with `CUDA 9`](https://github.com/weigao95/surfelwarp/blob/master/doc/ubuntu_pcl_build.md) (note that `CUDA 10` doesn't work with this one)
- [Ubuntu 16.04/18.04 with  `CUDA >= 9` (`CUDA 10` is OK) and `cilantro`](https://github.com/weigao95/surfelwarp/blob/master/doc/ubuntu_cliantro_build.md) 
- [Windows 10 with `CUDA 9`](https://github.com/weigao95/surfelwarp/blob/master/doc/windows_pcl_build.md) (`CUDA 10` doesn't work with this one)
- [Windows 10 with  `CUDA >= 9` (`CUDA 10` is OK) and `cilantro`](https://github.com/weigao95/surfelwarp/blob/master/doc/windows_cilantro_build.md) 

We also provide a [pre-built binary](https://github.com/weigao95/surfelwarp/tree/master/test_data/binary) for the windows platform (The CUDA  `-arch` flag for this executable is `sm_60`).

### Run Instructions

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

  You might override the [FetchInterface](https://github.com/weigao95/surfelwarp/blob/master/imgproc/frameio/FetchInterface.h) and use it with [ImageProcessor](https://github.com/weigao95/surfelwarp/blob/master/imgproc/ImageProcessor.h). If you don't have performance requirement, you can also convert your data into the format of [VolumeDeform](https://www.lgdv.tf.fau.de/publicationen/volumedeform-real-time-volumetric-non-rigid-reconstruction/).

- How to deal with topology change?

  Currently, only periodic reinitialization is implemented. To use it, set [this flag](https://github.com/weigao95/surfelwarp/blob/bfb2ccaac5b986fb42888f41624a275c1594e084/test_data/boxing_config.json#L11) in config file. More advanced criterion of reinitialization would come soon.

### TODO

The code is re-factored and improved from the repo of our RSS paper. There are some planned new features and some old code  need to be ported into this repository. Here is a list of TODOs:

- [x] Add sparse feature correspondence
- [x] Implement albedo reconstruction
- [ ] Rewrite automatic reinitialization detection for the new geometry and loss function. Currently, only periodic reinitialization is implemented
- [ ] Port the GPU kdtree querying (although the overall speed is already relatively fast)

### Contact

If you have any question or suggestion regarding this work, please send me an email.
