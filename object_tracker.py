from collections import deque
from imutils.video import VideoStream
import imutils 
import argparse
import numpy as np
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",help='path to video file')
ap.add_argument("-b","--buffer",default=64,help='size of list of pts')
args = vars(ap.parse_args())

pts = deque(maxlen=args['buffer'])

colorupper=(180,255,255)
colorlower=(82,9,127)
if not args.get("video",False):
	vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)
while True:
	frame=vs.read()
	frame = frame[1] if args.get("video",False) else frame
	if frame is None:
		break
	frame = imutils.resize(frame,width=600)
	blurred = cv2.GaussianBlur(frame,(11,11),0)
	hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)	
	mask = cv2.inRange(hsv,colorlower,colorupper)
	mask = cv2.erode(mask,None,iterations=2)
	mask = cv2.dilate(mask,None,iterations=2)
	cnts= cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	print(len(cnts))
	#print('area',cv2.contourArea(cnts[0]),'\n')
	if len(cnts)>0:	
		cnt = max(cnts,key=cv2.contourArea)
		((x,y),rad) = cv2.minEnclosingCircle(cnt)
		#center = (int(x),int(y))
		#(x,y) = (x-(w/2),y-(h/2))
		#box = cv2.boxPoints(rect)
		#box = np.int0(box)
		M = cv2.moments(cnt)
		center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
		#if w > 10 and h > 2:
		#	cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,255),2)
		#	cv2.circle(frame,center,2,(0,0,255),-1)
		if rad > 10:
			cv2.circle(frame,(int(x),int(y)),int(rad),(0,255,255),2)
			cv2.circle(frame,center,5,(0,0,255),-1)
	pts.appendleft(center)
	for i in range(1,len(pts)):
		if pts[i] is None or pts[i-1] is None:
			continue
		cv2.line(frame,pts[i-1],pts[i],(255,0,0),2)
	cv2.imshow('frame',frame)
	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break
if not args.get("video",False):
	vs.stop()
else:
	vs.release()
cv2.destroyAllWindows()		
