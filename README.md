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
