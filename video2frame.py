# read video and extract frames
# reference: https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/

import cv2

#extrace frames
def VideoToFrame(path):
	# use VideoCapture class
	# reference: https://docs.opencv.org/3.4/javadoc/org/opencv/videoio/VideoCapture.html
	video_object = cv2.VideoCapture(path)
	
	# check whether open the right video
	if video_object.isOpened():
		print("open sucessful")
	else:
		print("error!")

	count = 0
	#check whether frames were extracted
	isExtracted = 1

	while isExtracted:
		# read()
		isExtracted, image = video_object.read()
		cv2.imwrite("/Users/shanfandi/Desktop/frames/frame%d.jpg" % count, image)
		count += 1

VideoToFrame("/Users/shanfandi/Desktop/2019-10-10-14-07-25.bag/img_raw_03.avi")
print("finish")

