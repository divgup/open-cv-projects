import numpy as np
#import gradient
import bisect
import math
import cv2
#from skimage import feature
#import PIL.ImageDraw as ImageDraw,PIL.Image as Image, PIL.ImageShow as ImageShow
from skimage.draw import line
def gradients(image):
	#print(image)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	if gray.dtype.kind == 'u':
	# convert uint image to float
    # to avoid problems with subtracting unsigned numbers
		gray = gray.astype('float')

	# origsobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)	
	# origsobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
	# sobelx = np.abs(origsobelx)
	# sobelx_8u = np.uint8(sobelx)
	# sobely = np.abs(origsobely)
	# sobely_8u = np.uint8(origsobely)
	origsobely = np.empty(gray.shape, dtype=np.double)
	origsobelx = np.empty(gray.shape, dtype=np.double)
	origsobely[-1,:] = 0
	origsobely[0,:] = 0
	origsobely[1:-1,:] = gray[2:,:] - gray[:-2,:]
	origsobelx[:,1:-1] = gray[:,2:] - gray[:,:-2]
	origsobelx[:,-1]=0
	origsobelx[:,0]=0
	#print("abc")
	return origsobelx,origsobely

def angle_grad(image):	
	origsobelx,origsobely = gradients(image)
	#print(origsobelx[origsobely<0])
	ang = np.arctan(np.multiply(origsobely,1/(1e-5+origsobelx)))
	angle = np.rad2deg(np.arctan2(origsobely,origsobelx))%180
	#print(angle)
	#print(angle)
	#print(angle<0)
	ang[ang < 0]+= math.pi
	ang = ang*180/(math.pi)
	row,col = origsobelx.shape
	zipped_arr = np.zeros((row,col,2))
	zipped_arr = np.dstack((origsobelx,origsobely))
	grad_mag = np.zeros((row,col))
	#grad = np.zeros((origsobelx.shape))
	magnitude = np.hypot(origsobelx,origsobely)
	grad_mag = np.linalg.norm(zipped_arr,axis=2)
	#print(grad_mag)
	return angle,magnitude
def to_seq(window):
	return window.reshape(-1)
def normalize(b_size,method,orients_his_matrix,num_orients):
	stepy,stepx = b_size
	eps= 1e-5
	num_cell_rows = orients_his_matrix.shape[0]
	num_cell_col = orients_his_matrix.shape[1]
	block = np.zeros((num_cell_rows-stepy+1,num_cell_col-stepx+1,stepy*stepx*num_orients))
	for i in range(num_cell_rows-stepy+1):
		for j in range(num_cell_col-stepx+1):
			block[i,j] = orients_his_matrix[i:i+stepy,j:j+stepx].reshape(-1)	
			#block1 = orients_his_matrix[i:i+stepy,j:j+stepx,:]
			
			if method=='L1':
				block[i,j] = block[i,j]/(np.sum(np.abs(block[i,j]))+eps)	
				#block[i,j,:] = block1/(np.sum(np.abs(block1))+eps)	
			elif method=='L1-sqrt':
				block[i,j] = np.sqrt(block[i,j]/(np.sum(np.abs(block[i,j]))+eps))	
				#block[i,j,:] = np.sqrt(block1/(np.sum(np.abs(block1))+eps))	
			elif method=='L2':
				block[i,j] = block[i,j]/(np.sqrt(np.sum(block[i,j]**2))+eps)
				#block[i,j,:] = block1/(np.sqrt(np.sum(block1**2)+eps**2))
			#if (j==4 or j==5) and i==0:
			#	print(block[i,j].reshape(-1))	
	return block.reshape(-1)		
def hog_cell(sequence,orients,interval,num_orients):
	count = [0]*(num_orients)
	i=0
	#print('length',len(orients))
	for val in orients:
		if val==180:
			val = 0
		#else:	
		pos = bisect.bisect(interval,val)
		#print("pos-val",pos,'\t',val)
		if pos-1==len(interval):
			#print("wrong")
			continue
		#if pos==12:
		#	print('seq',sequence[i])	
		count[int(interval[pos-1]/int(180/num_orients))]+=sequence[i]
		i+=1	
	return np.array(count)/len(sequence)
def pyramid(img,scale=1.5,min_size=(30,30)):
	yield img
	h,w = img.shape[:2]
	#FPS = 1
	#fourcc = VideoWriter_fourcc(*'MP42')
	#video = VideoWriter('./pyramid.avi', fourcc, float(FPS), (width, height))
	while True:
		h/=scale
		w/=scale
		img = cv2.resize(img,(int(w),int(h)))
		if img.shape[0] < min_size[0] and img.shape[1] < min_size[1]:
			break
		yield img
def sliding_window(img,step,w_size):
	(h_w,w_w) = w_size
	i=0
	j=0
	height = img.shape[0]
	width = img.shape[1]	
	while i*step+h_w <=img.shape[0]: 
		while j*step+w_w <=img.shape[1]:
			yield((j*step,i*step),(j*step+w_w,i*step+h_w))
			j+=1
		j=0	
		i+=1
def visualize(img,orients_his_matrix,num_orients,c_row,c_col):
	s_row,s_col = img.shape[:2]
	#print(s_row,s_col)
	num_cell_rows,num_cell_col = int(img.shape[0]/c_row),int(img.shape[1]/c_col)
	radius = min(c_row,c_col)//2-1
	orients = np.arange(num_orients)
	bin_midpoints = np.pi*(orients+0.5)/num_orients
	dr_arr = radius*np.sin(bin_midpoints)		
	dc_arr = radius*np.cos(bin_midpoints)
	hog_image = np.zeros((s_row, s_col), dtype=float)
	# draw = ImageDraw.Draw(im)
	
	for r in range(num_cell_rows):
		for c in range(num_cell_col):
			for o,dr,dc in zip(orients,dr_arr,dc_arr):
				center = tuple([r*c_row+c_row//2,c*c_col+c_col//2])
				#rr, cc = line(int(center[0]+dr),int(center[1]-dc), int(center[0]-dr), int(center[1]+dc))
				rr, cc = line(int(center[0]-dc),int(center[1]+dr), int(center[0]+dc), int(center[1]-dr))
				hog_image[rr,cc]+=orients_his_matrix[r,c,o]
			#print(orients_his_matrix[r,c,11])
	return hog_image			
def hog(image,w_size,box_size,num_orients=12,method='L1'):
	#scale = 1.5
	step = w_size[0]
	# for image in pyramid(img,scale):
	num_cell_rows = image.shape[0]/step
	num_cell_col = image.shape[1]/step
	#print(image)
	orients,grad = angle_grad(image)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	interval = np.arange(0,180,180/num_orients)
	orients_his_matrix = np.zeros((int(num_cell_rows),int(num_cell_col),int(num_orients)))	
	for (x,y) in sliding_window(image,step,w_size):
		x_start,y_start = x
		x_end , y_end =  y	
		slidewindow_grad = grad[y_start:y_end,x_start:x_end]

		slidewindow_orients=orients[y_start:y_end,x_start:x_end]
		#if x_start==80 and x_end==96 and y_start==0:
			#print('slidewindow_grad',slidewindow_grad,'\n')
			#print(slidewindow_orients)
		seq = to_seq(slidewindow_grad)
		ang = to_seq(slidewindow_orients)
		orients_his = hog_cell(seq,ang,interval,num_orients)
		cell_col = x_start/step
		cell_row = y_start/step
		orients_his_matrix[int(cell_row),int(cell_col)] = orients_his

	hog_image = visualize(image,orients_his_matrix,num_orients,w_size[0],w_size[1])
	#H,himg = feature.hog(gray, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(2, 2), transform_sqrt=False, block_norm="L1",visualize=True)
# 	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")	
	#cv2.imshow('hog_image',hog_image)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#cv2.imshow('himg',himg)
	#cv2.waitKey(0)
	#diff = himg-hog_image
	#cv2.destroyAllWindows()
	normfeature = normalize(box_size,method,orients_his_matrix,num_orients)
	return normfeature,hog_image
# img = cv2.imread('ad1.jpg')
# img = cv2.resize(img,(64,128))
# myf,hogf,hog_image = hog(img,(16,16),(2,2))
# print(myf,'\t',hogf)
# diff = myf-hogf
# #diff_img = diffimg.reshape(-1)
# #print(diff_img[np.argmax(diff_img)],diff_img[np.argmin(diff_img)])
# #print(myf[240:260],hogf[255])
# print("max",diff[np.argmax(diff)],"min",diff[np.argmin(diff)])

