for i in *.wav; do ffmpeg -i "$i" -f segment -segment_time 1.2 -c copy "$i"%03d.wav; rm "$i"; done
ffprobe 1404_2055 Final.wav 2>&1 | grep -A1 Duration: