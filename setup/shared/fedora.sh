#!/usr/bin/env bash

# This script contains generic helpers and reusable functions and variables
# that will work on Fedora.



# == Helpers ==================================================================
log_header() { echo; echo ">>> $*"; }
log() { echo "[INFO]: $*"; }
warn() { echo "[WARN]: $*"; }
die() { echo "[FATAL]: $*" >&2; exit 1; }



# == OS Checks ================================================================
perform_os_check() {
    local required_version="${1:?Error: OS version is required for OS check.}"

    if [[ ! "$required_version" =~ ^-?[0-9]+$ ]]; then
        die "'$required_version' is not a valid integer." >&2
    fi

    . /etc/os-release
    [[ "${ID:-}" == fedora ]] || die "This script requires Fedora."
    [[ "${VERSION_ID:-}" == "${required_version}" ]] \
        || die "Requires Fedora ${required_version}; detected ${VERSION_ID:-unknown}."
}



# == Repositories =============================================================
install_rpm_fusion_free() {
    local current_version="${1:?Error: Must supply current OS version.}"

    local rpm_fusion_free_url="https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-${current_version}.noarch.rpm"

    if ! dnf repolist --enabled 2>/dev/null | grep -qi "rpmfusion-free"; then
        sudo dnf install -y "${rpm_fusion_free_url}" || die "Could not install RPM Fusion Free Repository for Fedora ${current_version}. Exiting."
        log "RPM Fusion Free Repository installed."
    else
        log "RPM Fusion Free Repository already present, skipping installation."
    fi
}

install_rpm_fusion_nonfree() {
    local current_version="${1:?Error: Must supply current OS version.}"

    local rpm_fusion_nonfree_url="https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-${current_version}.noarch.rpm"

    if ! dnf repolist --enabled 2>/dev/null | grep -qi "rpmfusion-nonfree"; then
        sudo dnf install -y "${rpm_fusion_nonfree_url}" || die "Could not install RPM Fusion Nonfree Repository for Fedora ${current_version}. Exiting."
        log "RPM Fusion Nonfree Repository installed."
    else
        log "RPM Fusion Nonfree Repository already present, skipping installation."
    fi
}

install_cuda_repo() {
    local cuda_fedora_ver="${1:?Error: Must supply CUDA repo Fedora version.}"

    local cuda_repo_id="cuda-fedora${cuda_fedora_ver}-x86_64"
    local cuda_repo_url="https://developer.download.nvidia.com/compute/cuda/repos/fedora${cuda_fedora_ver}/x86_64/cuda-fedora${cuda_fedora_ver}.repo"

    local repo_opts="${cuda_repo_id}.exclude="
    repo_opts+="nvidia-driver*,cuda-drivers*,nvidia-modprobe,nvidia-persistenced,"
    repo_opts+="nvidia-settings,nvidia-libXNVCtrl*,nvidia-xconfig,nvidia-open,nvidia-imex,"
    repo_opts+="nvidia-kmod-common,libnvidia-fbc,kmod-nvidia-*-dkms,"
    repo_opts+="nvidia-fs*,nvidia-gds*,xorg-x11-nvidia*"

    local cuda_repos
    cuda_repos="$(dnf repolist --all 2>/dev/null)" || die "Could not list DNF repositories."
    if ! grep -Fq "${cuda_repo_id}" <<< "${cuda_repos}"; then
        log "Adding NVIDIA CUDA repository for Fedora ${cuda_fedora_ver}..."
        sudo dnf config-manager addrepo --from-repofile "${cuda_repo_url}" \
            || die "Could not add repository for '${cuda_repo_id}' from '${cuda_repo_url}'."
    else
        log "NVIDIA CUDA repository is already configured."
    fi

    log "Ensuring NVIDIA CUDA repository is enabled..."
    sudo dnf config-manager setopt "${cuda_repo_id}.enabled=1" \
        || die "Could not enable '${cuda_repo_id}'."
    sudo dnf config-manager setopt "${repo_opts}" \
        || die "Could not set package exclusions for '${cuda_repo_id}'."
}



# == Dependencies =============================================================
dnf_cache_make_and_clean() {
    sudo dnf clean all
    sudo dnf makecache || die "dnf makecache failed after configuring CUDA repo"
}

install_nvidia_drivers() {
    if ! which nvidia-smi >/dev/null; then
        log "NVIDIA drivers not installed."

        sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda

        die "Driver packages installed. Reboot, verify nvidia-smi, then rerun this script. Please reboot system before continuing with the rest of the script."
    else
        log "NVIDIA drivers already present."
    fi
}

install_main_dependencies() {
    sudo dnf install -y \
        git \
        python3-devel \
        python3-numpy \
        cmake \
        ninja-build \
        gcc \
        gcc-c++ \
        make \
        gtk3-devel \
        libpng-devel \
        libjpeg-turbo-devel \
        || die "Something failed when installing main dependencies."
}

install_ffmpeg_dependencies() {
    sudo dnf install -y --allowerasing \
        ffmpeg \
        ffmpeg-devel \
        ffmpeg-libs \
        || die "Something failed when installing ffmpeg dependencies."
}



# == Build Tool & Dependency Checks ===========================================
perform_nvcc_check() {
    export PATH="${PATH}:/usr/local/cuda/bin"

    if ! command -v nvcc >/dev/null; then
        log "nvcc not present. Installing CUDA Toolkit..."
        sudo dnf install -y cuda-toolkit
        command -v nvcc >/dev/null || die "CUDA Toolkit installation did not provide nvcc."
        log "CUDA Toolkit installed."
    else
        log "nvcc present. Skipping CUDA Toolkit installation."
    fi
}

perform_video_codec_header_check() {
    local missing_headers=()
    local hdr
    for hdr in nvcuvid.h cuviddec.h nvEncodeAPI.h; do
        if [[ ! -f "/usr/local/cuda/include/${hdr}" ]]; then
            missing_headers+=("${hdr}")
        fi
    done

    if [[ ${#missing_headers[@]} -gt 0 ]]; then
        log "WARNING: The following headers were not found under /usr/local/cuda/include:"
        for hdr in "${missing_headers[@]}"; do
            log "  - ${hdr}"
        done
        log "Download the NVIDIA Video Codec SDK from:"
        log "  https://developer.nvidia.com/nvidia-video-codec-sdk"
        log "and copy the Interface/ headers to /usr/local/cuda/include/ before continuing."
        die "Missing NVENC/NVDEC headers. See instructions above."
    else
        log "All NVENC/NVDEC headers found."
    fi
}



# == OpenCV & OpenCV Contrib Installations ====================================
ensure_repo() {
    local dir="$1" url="$2"
    local clean_build="${3:?Error: Must supply clean build flag.}"
    local commit_want commit_have repo_changes

    if (( clean_build )); then
        log "Clean build enabled. Removing existing '${dir}' repo..."
        rm -rf -- "${dir}"
    fi

    if [[ ! -e "${dir}" ]]; then
        git init --quiet "${dir}" || die "Could not create ${dir}."
    elif [[ ! -d "${dir}/.git" ]]; then
        die "${dir} exists but is not a Git checkout; inspect it before proceeding."
    else
        repo_changes="$(git -C "${dir}" status --porcelain)" || die "git status failed in ${dir}."
        [[ -z "${repo_changes}" ]] || die "${dir} has local changes; commit or remove them first."
    fi

    # Fetch the tag into FETCH_HEAD. This doesn't create a local tag.
    # Basically ensures we get the 'exact' version we want, without it being finicky about
    # tags or commit hashes and whatnot.
    git -C "${dir}" fetch --depth 1 --no-tags "${url}" "refs/tags/${OPENCV_VERSION}" \
        || die "Could not fetch tag ${OPENCV_VERSION} from ${url}."

    # ^{commit} turns an annotated tag object into the commit it points to.
    # Tags have separate hashes compared to the commits, so this is saying
    # "follow this tag until you get to the commit it belongs to."
    commit_want="$(git -C "${dir}" rev-parse --verify "FETCH_HEAD^{commit}")" \
        || die "Could not resolve tag ${OPENCV_VERSION} to a commit in ${dir}."

    # And finally check out what bit we do want indeed here.
    git -C "${dir}" checkout --quiet --detach "${commit_want}" \
        || die "Could not check out ${OPENCV_VERSION} in ${dir}."

    # Confirm our HEAD is the commit we fetched from upstream.
    commit_have="$(git -C "${dir}" rev-parse --verify HEAD)" || die "Could not read HEAD in ${dir}."
    if [[ "${commit_want}" != "${commit_have}" ]]; then
        die "${dir}: expected commit ${commit_want}, got ${commit_have}."
    fi

    log "${dir##*/}: ${OPENCV_VERSION} (${commit_have})"
}

verify_cmake_flags() {
    local config_dir="${1:?Error: must provide a source OpenCV Contrib directory.}"

    # These map to well-known CMakeCache keys; adjust if upstream renames them.
    local feature
    for feature in HAVE_CUDA HAVE_CUBLAS HAVE_NVCUVID HAVE_NVCUVENC; do
        grep -Eq "^#define ${feature}([[:space:]]|$)" "${config_dir}/cvconfig.h" \
            || die "CMake did not enable '${feature}'. Check the config output above."
        log "  OK ${feature}"
    done
}



# == Post-Install Functions ===================================================
create_and_update_ldconfig() {
    local opencv_dir_prefix="${1:?Error: OpenCV directory prefix is needed.}"

    local ldconf_file="/etc/ld.so.conf.d/opencv.conf"

    sudo tee "${ldconf_file}" > /dev/null <<EOF
# Added by build_opencv_fedora_43
${opencv_dir_prefix}/lib64
EOF

    sudo ldconfig
    log "ldconfig updated. File written to '${ldconf_file}'."
}

perform_python_sanity_check() {
    local opencv_ver="${1:?Error: OpenCV version is needed.}"
    local opencv_dir_prefix="${2:?Error: OpenCV directory prefix is needed.}"

    local cv2_init_py py_site_pkgs
    cv2_init_py="$(find "${opencv_dir_prefix}" -type f -path '*-packages/cv2/__init__.py' -print -quit)"

    if [[ -z "${cv2_init_py}" ]]; then
        die "Python bindings not found under ${opencv_dir_prefix}."
    fi

    py_site_pkgs="$(dirname "$(dirname "${cv2_init_py}")")"

    PYTHONPATH="${py_site_pkgs}" EXPECTED="${opencv_ver}" python3 - <<'PYEOF'
import os, re, cv2

expected = os.environ["EXPECTED"]
if cv2.__version__ != expected:
    raise SystemExit(f"Loaded OpenCV {cv2.__version__} from file {cv2.__file__}, expected {expected}.")
if not re.search(r"FFMPEG:\s+YES", cv2.getBuildInformation()):
    raise SystemExit(f"OpenCV was built without FFmpeg support. Please investigate build logs and try again.")

count = cv2.cuda.getCudaEnabledDeviceCount()
print(f"CUDA-enabled devices visible to OpenCV: {count}")
if count == 0:
    raise SystemExit("No CUDA devices found - check driver and CUDA installation.")
build_info = cv2.getBuildInformation()
for keyword in ("CUDA", "NVCUVID", "NVCUVENC", "FFmpeg"):
    for line in build_info.splitlines():
        if keyword in line:
            print(line.strip())
            break
PYEOF
}
