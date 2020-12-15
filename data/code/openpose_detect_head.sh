# work on frame.jpg
cd openpose && ./build/examples/openpose/openpose.bin --image_dir '{frame_folder}' --render_pose 0 --display 0 --write_json '{json_dir}'

#get render image
cd openpose && ./build/examples/openpose/openpose.bin --image_dir '{colab_img_path}' --face --display 0 --write_images '{colab_openpose_image_save_path}' # --net_resolution "-1x736" --scale_number 4 --scale_gap 0.25
