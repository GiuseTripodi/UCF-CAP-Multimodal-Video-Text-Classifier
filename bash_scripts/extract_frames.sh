mkdir /Users/user/PycharmProjects/frozen-in-time/data/UcfCap/UCF_frames
for video in /Users/user/PycharmProjects/frozen-in-time/data/UcfCap/YouTubeClips/*.mp4; do
    folder=/Users/user/PycharmProjects/frozen-in-time/data/UcfCap/YouTubeClips/$(basename "$video" .mp4)
    mkdir -p "$folder"
    ffmpeg -i "$video" "$folder/%04d.jpg"
done
