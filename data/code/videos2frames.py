# read video and extract frames
# reference: https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/

import cv2
import os

#extrace frames
def VideoToFrame(path):
	# use VideoCapture class
	# reference: https://docs.opencv.org/3.4/javadoc/org/opencv/videoio/VideoCapture.html
	video_object = cv2.VideoCapture(path)
	
	# check whether open the right video
	if video_object.isOpened():
		print("--- open sucessful ---")
	else:
		print("--- error ---!")

	count = 0
	#check whether frames were extracted
	isExtracted = 1

	while isExtracted:
		# read()
		isExtracted, image = video_object.read()
		cv2.imwrite("./frames/frame%d.jpg" % count, image)
		count += 1

	print("Finish!")

# make frame dir for each video
def mkdir(path):
	folder = os.path.exists(path)

	if not folder:
		os.makedirs(path)
		print('--- create new folder successful---')
	else:
		print("--- Folder already exists! ---")

# go to data folder
os.chdir('data')

video_folder = os.listdir(os.getcwd())
print(video_folder)

for each_folder in video_folder:
	#ignore .DS_Store
	if each_folder != ".DS_Store":

		# go to target bag
		os.chdir(each_folder)
		current_path = os.getcwd()
		print("current path: " + current_path)

		# create frame folder
		file = current_path + "/frames"
		mkdir(file)

		# get video path
		video_path = current_path + "/img_raw_03.avi"
		# extract videos into frames
		VideoToFrame(video_path)

		# back to data folder
		os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))

