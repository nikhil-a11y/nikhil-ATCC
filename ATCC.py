import numpy as np
import argparse
import cv2
import subprocess
import datetime
import shutil, os
import random as rn
import time
from yolo_utils import infer_image, show_image
import pyodbc as db
from imutils.video import FPS
from imutils.video import VideoStream


FLAGS = []

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov4-tiny-custom_40000.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov4-tiny-custom.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.8,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.8')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.5,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')

	FLAGS, unparsed = parser.parse_known_args()

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
	 # Configure the network backend
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# If both image and video files are given then raise error
	print ('Neither path to an image or path to video provided')
	print ('Starting Inference on Webcam')


	# Infer real-time on webcam
	count = 0
	#url= 'rtsp://root:root@123@192.168.60.41:554/live1s1.sdp'
	url= 'rtsp://admin:Iis@1234@192.168.0.64:554/Streaming/channels/001/?transportmode=unicast'
	vid = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
	font = cv2.FONT_HERSHEY_PLAIN
	#add time
	st_time = time.time()
	frame_id=0
	date=datetime.datetime.now().strftime("%Y-%m-%d")
	garbage = "C:\\FTP\\ALV\\L4\\"
	if os.path.exists(garbage):
		test = os.listdir(garbage)
		if test:
			for i in test:
				if i.endswith('.avi'):
					os.remove(os.path.join(garbage,i))
	else:
		os.makedirs(garbage)
	#conx = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-ABHSMVB;DATABASE=IIS;TRUSTED_CONNECTION=YES"
	#con = db.connect(conx)
	#cur = con.cursor()
	Tid= "C:\\FTP\\Tid\\L4\\"
	if os.path.exists(Tid):
		test = os.listdir(Tid)
		if test:
			for i in test:
				if i.endswith('.txt'):
					os.remove(os.path.join(Tid,i))
	else:
		os.makedirs(Tid)
	while vid.isOpened():
		_, frame = vid.read()
		x = datetime.datetime.now()
		AVC_date=x.strftime("%Y-%m-%d")
		class_date=x.strftime("%Y%m%d")
		class_time=x.strftime("%H%M%S")
		directory = "C:\\FTP\\AVC\\L4\\" + AVC_date
		if not os.path.exists(directory):
			os.makedirs(directory)
		#fps count
		frame_id+=1
		height, width = frame.shape[:2]
		if count == 0:
			frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
	    						height, width, frame, colors, labels, FLAGS)
			count += 1

		else:
			frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
	    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
			count = (count + 1) % 6
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		end_time= time.time() - st_time
		fps= frame_id/end_time
		cv2.putText(frame,"FPS :"+str(fps),(60,300),font,3,(0,0,0),10)
		cv2.imshow('YOLOv4_4', cv2.resize(frame, (720, 420)))
		#T_file = os.listdir(Tid)
		#k=cv.waitKey(1)
		if boxes :
			img_name = class_date +'_'+ class_time +'_' + labels[classids[0]]+"_.png"
			if not os.path.exists(garbage + img_name):
				cv2.imwrite(garbage + img_name, gray)
				print(boxes)
		elif len(boxes)==0:
			test = os.listdir(garbage)
			if test:
				rm=rn.choice(test)
				shutil.copy(garbage + rm , directory)
				ls=rm.split('_')
				print(ls)
				# cur.execute("INSERT INTO AVC_Class VALUES(?,?,?,?,?,?,?)", (ls[2],ls[0],ls[1],'ok','1','No','L4'))
				# con.commit()
				for i in test:
					if i.endswith('.png'):
						os.remove(os.path.join(garbage,i))
			#for i in T_file:
			#	if i.endswith('.txt'):
			#		os.remove(os.path.join(T_file,i))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	#database connection
		#break
	vid.release()
	cv2.destroyAllWindows()
