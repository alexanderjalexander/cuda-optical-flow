#!/usr/bin/env bash

# Source - https://stackoverflow.com/a/246128
# Posted by dogbane, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-29, License - CC BY-SA 4.0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ $(which ffmpeg 2>/dev/null) ]]; then
    exit 1
fi

mkdir -p giraffes

trap 'echo "Stopping..."; exit' INT

ffmpeg -y -i "${SCRIPT_DIR}/inputs/Giraffe.mp4" -c:v libx264 -crf 18 -preset slow -vf "scale=iw/4:ih/4,trim=start_frame=167,setpts=PTS-STARTPTS" "${SCRIPT_DIR}/giraffes/Giraffe-540p-100f.mp4"
ffmpeg -y -i "${SCRIPT_DIR}/inputs/Giraffe.mp4" -c:v libx264 -crf 18 -preset slow -vf "scale=iw/3:ih/3,trim=start_frame=167,setpts=PTS-STARTPTS" "${SCRIPT_DIR}/giraffes/Giraffe-720p-100f.mp4"
ffmpeg -y -i "${SCRIPT_DIR}/inputs/Giraffe.mp4" -c:v libx264 -crf 18 -preset slow -vf "scale=iw/2:ih/2,trim=start_frame=167,setpts=PTS-STARTPTS" "${SCRIPT_DIR}/giraffes/Giraffe-1080p-100f.mp4"
ffmpeg -y -i "${SCRIPT_DIR}/inputs/Giraffe.mp4" -c:v libx264 -crf 18 -preset slow -vf "scale=iw/1.5:ih/1.5,trim=start_frame=167,setpts=PTS-STARTPTS" "${SCRIPT_DIR}/giraffes/Giraffe-1440p-100f.mp4"
ffmpeg -y -i "${SCRIPT_DIR}/inputs/Giraffe.mp4" -c:v libx264 -crf 18 -preset slow -vf "scale=iw*1:ih*1,trim=start_frame=167,setpts=PTS-STARTPTS" "${SCRIPT_DIR}/giraffes/Giraffe-4k-100f.mp4"
ffmpeg -y -i "${SCRIPT_DIR}/inputs/Giraffe.mp4" -c:v libx264 -crf 18 -preset slow -vf "scale=iw*1.5:ih*1.5,trim=start_frame=167,setpts=PTS-STARTPTS" "${SCRIPT_DIR}/giraffes/Giraffe-6k-100f.mp4"
ffmpeg -y -i "${SCRIPT_DIR}/inputs/Giraffe.mp4" -c:v libx264 -crf 18 -preset slow -vf "scale=iw*2:ih*2,trim=start_frame=167,setpts=PTS-STARTPTS" "${SCRIPT_DIR}/giraffes/Giraffe-8k-100f.mp4"
