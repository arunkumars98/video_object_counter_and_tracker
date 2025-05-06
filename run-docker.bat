docker build -t video_object_counter_and_tracker .

# Run the container
docker run --device=/dev/video0:/dev/video0 -p 5000:5000 video_object_counter_and_tracker
