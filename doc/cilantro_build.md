### Building cilantro and Pangolin via terminal

Here is the complete set of instructions.

#### Eigen
First, make sure `Eigen` is properly installed.
To install, open the terminal and run:
`sudo apt-get install libeigen3-dev`
Alternatively you can follow the standard cmake procedure and build and install the `Eigen 3.3.9` version included in `external/eigen3.4`. Note that running `make install` (or `sudo make install` if installing system-wide) is critical to generate the _Eigen3Config.cmake_ file needed by `cilantro`.

#### Pangolin
If you have SSH GitHub access:

`git clone git@github.com:stevenlovegrove/Pangolin.git`

\------------- OR via HTTPS: ----------------

`git clone https://github.com/stevenlovegrove/Pangolin.git` 

Then:

- `cd Pangolin`
- `mkdir build_static && mkdir install_static`
- `cd build_static`

If the desired `Eigen` is installed system-wide, run:
 
 - `cmake -DCMAKE_INSTALL_PREFIX=../install_static -BUILD_SHARED_LIBRARIES=OFF ..`
 
 Otherwise, append specification of Eigen's include folder, i.e.:
 `-DEIGEN_INCLUDE_DIR=<directory from which the Eigen/Eigen header can be accessed>`.
 
Now, run `make -jX` with appropriate number of cores, and install, e.g.:

- `make -j8 && make install`

And return to top-level directory:

- `cd ../..`
#### cilantro

Please use this commit `e9eb9f3c5c75710eb6cfeeb6313b5a73aaa06a28`. The commands are (adjust for your platform as necessary):

- `git clone git@github.com:kzampog/cilantro.git`
- `cd cilantro && git checkout e9eb9f3c5c75710eb6cfeeb6313b5a73aaa06a28`
- `mkdir build_static && mkdir install_static`
- `cmake -DCMAKE_INSTALL_PREFIX=../install_static -DBUILD_SHARED_LIBRARIES=OFF -DPangolin_DIR=../Pangolin/install_static/lib/cmake/Pangolin ..`
(If the desired Eigen is in a non-default location, also add:  `-DEigen3_DIR=<path to Eigen3Config.cmake, e.g. external/eigen3.4/install/share/eigen3/cmake>`)

- `make -j4 && make install`
- `cd ../..`

Due to a bug in cilantro CMake system (at time of writing), you'll have to specify **both** `Pangolin_DIR` and `cilantro_DIR` when building `SurfelWarp`.
