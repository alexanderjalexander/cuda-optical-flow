# cuda-optical-flow

This repo serves to host the files for a CUDA-based implementation of Sparse Lucas-Kanade, aiming to leverage GPU power in order to accelerate optical flow.

## Development

### Dependency Installation on Fedora 43

These instructions assume you have a qualifying NVIDIA GPU with architecture at least SM_75.

+ Install the `opencv` development headers
  - `sudo dnf install opencv opencv-devel`
+ Install CUDA Toolkit
  - See the [official instructions here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Fedora&target_version=43&target_type=rpm_network) for installing it over a network installation.
+ Clone the repository