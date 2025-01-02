#!/bin/bash

# Path to the ffmpeg binary
FFMPEG_PATH="/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/bash_scripts/ffmpeg-7.0.2-amd64-static/ffmpeg"

# Create the output folder
mkdir -p /Users/user/PycharmProjects/frozen-in-time/data/UcfCap/UCF_frames

# Loop through the videos and extract frames
for video in /mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data/UcfCap/YouTubeClips/*.mp4; do
    folder=/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data/UcfCap/YouTubeClips/$(basename "$video" .mp4)
    mkdir -p "$folder"

    # Call ffmpeg using the full path to the binary
    $FFMPEG_PATH -i "$video" "$folder/%04d.jpg"
done

