

# cuda-optical-flow

Este repositorio sirve para alojar los archivos de una implementación basada en CUDA de Sparse Lucas-Kanade, con el objetivo de aprovechar la potencia de la GPU para acelerar el flujo óptico.

## Desarrollo

### Instalación de dependencias en Fedora 43

Estas instrucciones asumen que cuenta con una GPU NVIDIA compatible con una arquitectura de al menos SM_75.

+ Instale los controladores NVIDIA adecuados, ya sea desde cuda-fedora43-... o desde los repositorios RPM nonfree.
+ Instale las cabeceras de desarrollo de `opencv`
  - `sudo dnf install opencv opencv-devel`
+ Instale CUDA Toolkit
  - Consulte las [instrucciones oficiales aquí](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Fedora&target_version=43&target_type=rpm_network) para realizar una instalación por red.
+ Clone el repositorio

### Instalación de dependencias en Debian 13

+ Asegúrese de habilitar los repositorios `contrib`, `non-free` y `non-free-firmware` editando `/etc/apt/sources.list`
+ Instale y ejecute `nvidia-detect` para encontrar el controlador NVIDIA específico que funcione con su sistema.
+ Instale dicho controlador, junto con `nvidia-kernel-dkms`, `linux-headers-$(uname -r)`, `build-essential` y, finalmente, `dkms`
+ Instale `libopencv-dev`
+ Instale CUDA Toolkit 13.2 según las instrucciones en [este enlace](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=13&target_type=deb_network)

# Créditos de las fuentes

## Videos de entrada

Varios videos de entrada aquí fueron utilizados como parte de las pruebas y la evaluación de este programa.

Algunos videos se convirtieron a mp4 con codificación H264 y CRF 18, con el fin de almacenarlos en este repositorio de GitHub.

- Slow-Traffic = OpenCV
  - https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
- Demon-Slayer.mp4 = OpenCV
  - https://opencv.org/wp-content/uploads/2025/02/Example-Video.mp4
- Plane-Dock.mp4 = Jason Hendardy
  - https://www.dpreview.com/videos/5805014353/panasonic-gh6-video-sample-gallery
