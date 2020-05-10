import cv2
import numpy as np
from pyimagesearch.transform import four_point_transform
import imutils
from skimage.filters import threshold_local
from skimage import measure
image = cv2.imread('images/test_01.png')
orig = image.copy()
orig = imutils.resize(orig,height=500)
ratio = image.shape[0]/500
image = imutils.resize(orig,height = 500)

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image,(5,5),0)
edged_image = cv2.Canny(gray_image,75,200)
cnts = cv2.findContours(edged_image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print(cnts)
#print('cnts=',cnts)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
#print(np.array(cnts[0]).shape)

for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
cv2.drawContours(image,[screenCnt],0,(0,255,0),2)
#cv2.imshow('contour',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
orig = four_point_transform(orig,screenCnt.reshape(4,2))
warped = four_point_transform(gray_image, screenCnt.reshape(4, 2))
cv2.imshow('omr',orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#cv2.imshow('thresh',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
circleCnt=[]
for cnt in cnts:
	(x,y,w,h) = cv2.boundingRect(cnt)
	ar = w/(float(h))
	if w>20 and h>20 and ar > 0.9 and ar<1.1:
		circleCnt.append(cnt)
#print(circleCnt)
copy = orig.copy()
centers = []
#print(len(circleCnt))
for cnt in circleCnt:
	x = cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00']
	y = cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00']
	centers.append((y,x))

#centers.sort()
#print('centers',len(centers),'\n')
#for cnt in circleCnt:
sorted_y = sorted(circleCnt,key=lambda x:int(cv2.moments(x)['m01']/cv2.moments(x)['m00']))
#print(sorted_y[0:5])
#cv2.drawContours(copy,sorted_y,-1,(255,0,0),3)
#cv2.imshow('orig',copy)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
x=[]
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
#print(sorted_y)
sorted_x = []
correct=0
for i in range(5):
	sorted_x = sorted(sorted_y[i*5:5*(i+1)],key=lambda x:int(cv2.moments(x)['m10']/cv2.moments(x)['m00']))
	max_black_pix = 0
	for j in range(5):
		mask = np.zeros(thresh.shape,dtype='uint8')
		cv2.drawContours(mask,sorted_x,j,255,-1)
		bubble_fg = cv2.bitwise_and(thresh,thresh,mask = mask)
		#cv2.imshow('foreground',bubble_fg)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		#orig_fg = cv2.cvtColor(orig_fg,cv2.COLOR_BGR2GRAY)
		#cv2.imshow('orig_fg',orig_fg)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		#thres = cv2.threshold(orig_fg,150,255,cv2.THRESH_BINARY)[1]
		#cv2.imshow('thres',thres)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		black_pix = cv2.countNonZero(bubble_fg)
		if(black_pix > max_black_pix):
			max_black_pix = black_pix
			index = j	
	print('marked ans',index)		
	if(index==ANSWER_KEY[i]):
		correct+=1
print('correct',correct,'\n')
print('percentage correct',correct*100/5)		
#print(sorted_x[0:5])
#x = np.array(x)
#print(x.dtype)
#rint(sorted_x[0])
#mask = np.zeros(thresh.shape,dtype='uint8')
#cv2.drawContours(orig,,-1,255,-1)
#warped = four_point_transform(gray_image, screenCnt.reshape(4, 2))
#cv2.drawContours(orig,x,0,(255,0,0),3)
#cv2.imshow('orig',orig)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
		
# labels = measure.label(mask,neighbors=8,background=0)
# print(labels)
# mask = np.zeros(thresh.shape,dtype='uint8')
# #mythresh = 600
# for label in np.unique(labels):
# 	if label==0:
# 		continue
# 	labelMask = np.zeros(thresh.shape,dtype='uint8')
# 	labelMask[labels==label] = 255	
# 	num_pix = cv2.countNonZero(labelMask)
# 	if num_pix > 500 and num_pix < 750:
# 		mask = cv2.add(mask,labelMask)
# cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cnts=imutils.grab_contours(cnts)
# #ind = 1
# #for cnt in cnts:
# 	#M = cv2.moments(cnt)
# 	#cx = int(M['m10']/M['m00'])
# 	#cy = int(M['m01']/M['m00'])
# 	#centers.append([cy,ind])
# 	#ind+=1
# cnts = sorted(cnts,key=lambda x:int(cv2.moments(x)['m01']/cv2.moments(x)['m00']))
# print(cnts)
# #def compare_answers():

# cv2.drawContours(orig,cnts,-1,(255,0,0),3)
# cv2.imshow('orig',orig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
