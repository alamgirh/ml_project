# reference: read json file in python. https://blog.csdn.net/leviopku/article/details/103773219
import json
import math
import cv2
import os

# make dir for cropped head directory
def mkdir(path):
	folder = os.path.exists(path)

	if not folder:
		os.makedirs(path)
		print(path)
		print('--- create new folder successful---')
	else:
		print(path)
		print("--- Folder already exists! ---")

# extract neck/eyes/ears position from json file, then use neck/ears position to compute the head location, return x,y coor for head
# sometimes the robot is also detected as a human, use eyes/ears position to ignore the robot
def read_json(filename):
	f = open(filename,'r')
	content = f.read()
	# save json file as a 2d dict
	people_dict = json.loads(content)
	#print(type(people_dict))
	#res = not people_dict['people']
	if len(people_dict['people']) == 0:
		start_x = 0
		end_x = 0
		start_y = 0
		end_y = 0
	else:
		# pose keypoints for human
		target_position = people_dict['people'][0]['pose_keypoints_2d']

		# extract neck/eyes/ears position from json file
		neck_x = math.ceil(target_position[4])
		neck_y = math.ceil(target_position[3])
		right_eye_x = math.ceil(target_position[46])
		right_eye_y = math.ceil(target_position[45])
		left_eye_x = math.ceil(target_position[49])
		left_eye_y = math.ceil(target_position[48])
		left_ear_x = math.ceil(target_position[55])
		left_ear_y = math.ceil(target_position[54])
		right_ear_x = math.ceil(target_position[52])
		right_ear_y = math.ceil(target_position[51])
		# only detect right ear
		if left_ear_x == 0 and left_ear_y == 0:
			start_x = right_ear_x - 60
			end_x = right_ear_x + 60
			start_y = right_ear_y - 60
			end_y = right_ear_y + 60
		# only detect left ear
		elif right_ear_x ==0 and right_ear_y == 0:
			start_x = left_ear_x - 60
			end_x = left_ear_x + 60
			start_y = left_ear_y - 60
			end_y = left_ear_y + 60
		#ignore the robot, use the eyes & ears position
		elif (354 <= right_eye_x <= 358 or 360 <= right_eye_y <= 365) and (354 <= left_eye_x <= 358 or 365 <= left_eye_y <= 368) and (356 <= right_ear_x <= 360 or 357 <= right_ear_y <= 360) and (359 <= left_ear_x <= 363 or  359 <= left_ear_y <= 367):
			start_x = 0
			end_x = 0
			start_y = 0
			end_y = 0
		# both right and left ear are detected
		else:
			center_x = round((left_ear_x + right_ear_x) / 2)
			center_y = round((left_ear_y + right_ear_y) / 2)
			start_x = center_x - 60
			end_x = center_x + 60
			start_y = center_y - 60
			end_y =center_y + 60
	return (start_x, end_x, start_y, end_y)

datasets_path = '/content/drive/My Drive/open_pose/data'
dataset = os.listdir(datasets_path)
for each_dataset in dataset:
	# go to target bag
	dataset_path = datasets_path + '/' + each_dataset
	#make cropped img directory
	cropped_head_dir = dataset_path + '/heads'
	mkdir(cropped_head_dir)

	json_dir = dataset_path + '/json_file'
	frame_dir = dataset_path + '/frames'
	
	jsons = os.listdir(json_dir)
	for json_file in jsons:
		# name each head image, frame(number)_head.jpg
		cur_path = json_dir + '/' + json_file
		frame_name = cur_path.rsplit('/',1)[-1][:-15]
		frame = frame_dir + '/' + frame_name+'.jpg'

		# get x,y coor for head
		x_y = read_json(cur_path)
		start_x = x_y[0]
		end_x = x_y[1]
		start_y = x_y[2]
		end_y = x_y[3]
		if start_x != 0 and end_x != 0 and start_y != 0 and end_y != 0:
			img = cv2.imread(frame)
			# crop head
			crop_img = img[start_x:end_x, start_y:end_y]
			if crop_img.size != 0:
				cropped_head_path = cropped_head_dir + '/' + frame_name + '_head.jpg'
				cv2.imwrite(cropped_head_path, crop_img)
	
	print(dataset_path + '--- finished ---')


