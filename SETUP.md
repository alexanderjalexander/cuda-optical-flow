# OpenCV CUDA build - setup guide

## Fedora 43

### Prerequisites

You will need:

- An NVIDIA GPU (this guide targets `CUDA_ARCH_BIN=7.5`, i.e. RTX 20-series - adjust if yours differs)
- NVIDIA drivers already installed and working (`nvidia-smi` should return output)
- `opencv` and `opencv_contrib` source trees cloned side by side in the same directory as the build script

```
your-dir/
|- build_opencv_fedora
|- opencv/
|- opencv_contrib/
|- opencv_build/  <- created automatically by the script
```

---

### Step 0 - Clone OpenCV & OpenCV contrib

If you don't already have them installed, run the following commands:

```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

Both must be cloned into the same directory as the build script.

---

### Step 1 - CUDA repository

NVIDIA has a native Fedora 43 CUDA repo. If you don't have it yet, run this:

```bash
FEDORA_VER=$(rpm -E '%fedora')
sudo dnf config-manager addrepo --from-repofile \
    "https://developer.download.nvidia.com/compute/cuda/repos/fedora${FEDORA_VER}/x86_64/cuda-fedora43.repo"
sudo dnf clean all
sudo dnf makecache
```

If the repo is already present, skip this - the build script detects it automatically.

---

### Step 2 - RPM Fusion Free

Required for `ffmpeg` and `ffmpeg-devel`:

```bash
FEDORA_VER=$(rpm -E '%fedora')
sudo dnf install -y \
  https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-${FEDORA_VER}.noarch.rpm
```

---

### Step 3 - Install dependencies

```bash
sudo dnf install -y \
    python3-devel \
    python3-numpy \
    cmake \
    ninja-build \
    gcc-c++ \
    make \
    gtk3-devel \
    libpng-devel \
    libjpeg-turbo-devel \
    cuda-toolkit-13-2
```

FFmpeg must be installed separately with `--allowerasing`. Fedora versions 41 and up ship their own codec
packages (`libswscale-free`, `libswresample-free`, `libpostproc-free`) that conflict with `ffmpeg-libs`.

```bash
sudo dnf install -y --allowerasing \
    ffmpeg \
    ffmpeg-devel \
    ffmpeg-libs
```

---

### Step 4 - NVENC / NVDEC headers

The Video Codec SDK headers (`nvcuvid.h`, `cuviddec.h`, `nvEncodeAPI.h`) are required for hardware video
encode/decode support. They ship inside `cuda-toolkit-13-2` and should already be present at `/usr/local/cuda/include/`.

Verify:

```bash
ls /usr/local/cuda-13.2/include/nvcuvid.h \
   /usr/local/cuda-13.2/include/cuviddec.h \
   /usr/local/cuda-13.2/include/nvEncodeAPI.h
```

If any are missing, download the NVIDIA Video Codec SDK from https://developer.nvidia.com/nvidia-video-codec-sdk a
nd copy the headers inside the `Interface/` folder into `/usr/local/cuda-13.2/include/`.

---

### Step 5 - Run the build script

The script `setup/build_opencv_fedora_43` will automatically install the necessary dependencies for building this on Fedora 43.

You can run it in a Tmux session and let it do it's job with the following commands:
```bash
tmux new -s opencv
chmod +x build_opencv_fedora_43
./build_opencv_fedora_43
```

The script will:

1. Configure OpenCV with CMake and verify the flags before building
2. Build with Ninja using all available cores
3. Install to `/usr/local`
4. Write `/etc/ld.so.conf.d/opencv.conf` to register `/usr/local/lib64` with the dynamic linker (required
   on Fedora as this path is not searched by default)
5. Run a sanity check via Python to confirm CUDA devices are visible

---

### Step 6 - Verify

```bash
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

Should print `1` or higher. If it prints `0`, check your driver installation with `nvidia-smi`.