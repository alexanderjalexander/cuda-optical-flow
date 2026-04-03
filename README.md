# cuda-optical-flow

This repo serves to host the files for a CUDA-based implementation of Sparse Lucas-Kanade, aiming to leverage GPU power in order to accelerate optical flow.

## Development

### Dependency Installation on Fedora 43

These instructions assume you have a qualifying NVIDIA GPU with architecture at least SM_75.

+ Install the appropriate NVIDIA Drivers, either from cuda-fedora43-... or the RPM nonfree repositories.
+ Install the `opencv` development headers
  - `sudo dnf install opencv opencv-devel`
+ Install CUDA Toolkit
  - See the [official instructions here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Fedora&target_version=43&target_type=rpm_network) for installing it over a network installation.
+ Clone the repository

### Dependency Installation on Debian 13

+ Ensure that you enable the `contrib`, `non-free`, and the `non-free-firmware` repositories by editing `/etc/apt/sources.list`
+ Install and run `nvidia-detect` to find the specific NVIDIA driver that works for your system.
+ Install that driver, as well as `nvidia-kernel-dkms`, `linux-headers-$(uname -r)`, `build-essential`, and then `dkms`
+ Install `libopencv-dev`
+ Install CUDA Toolkit 13.2 according to the instructions at [this link here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=13&target_type=deb_network)

# Source Credits

## Video Inputs

Various video inputs here were used as part of the testing and evaluation for this program.

Some videos were converted to H264 mp4 with CRF 18, in order to save them onto this GitHub repository.

- Slow-Traffic = OpenCV
  - https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
- Demon-Slayer.mp4 = OpenCV
  - https://opencv.org/wp-content/uploads/2025/02/Example-Video.mp4
- Plane-Dock.mp4 = Jason Hendardy
  - https://www.dpreview.com/videos/5805014353/panasonic-gh6-video-sample-gallery
