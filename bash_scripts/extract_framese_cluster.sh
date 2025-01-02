mkdir /Users/user/PycharmProjects/frozen-in-time/data/UcfCap/UCF_frames
export PATH=$PATH:/path/to/extracted/ffmpeg
for video in /mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data/UcfCap/YouTubeClips/*.mp4; do
    folder=/mnt/iusers01/mace01/t08341gt/UCF_cap_mh/data/UcfCap/YouTubeClips/$(basename "$video" .mp4)
    mkdir -p "$folder"
    ffmpeg -i "$video" "$folder/%04d.jpg"
done
