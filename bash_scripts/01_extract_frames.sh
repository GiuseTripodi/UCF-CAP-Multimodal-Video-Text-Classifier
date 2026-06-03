#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIDEO_DIR="${1:-${PROJECT_ROOT}/data/UcfCap/YouTubeClips}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found in PATH" >&2
    exit 1
fi

for video in "${VIDEO_DIR}"/*.mp4; do
    folder="${video%.mp4}"
    mkdir -p "$folder"
    ffmpeg -i "$video" "$folder/%04d.jpg"
done
