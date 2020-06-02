import nms
def val_iou(gt,bbox):
    overlap_coord = [np.maximum(bbox[0],gt[0]),np.maximum(bbox[1],gt[1]), np.minimum(bbox[2],gt[2]) ,np.minimum(bbox[3],gt[3])]
    if overlap_coord[2]-overlap_coord[0]<0 or overlap_coord[3]-overlap_coord[1]<0:
        return 0
    overlap_area = (overlap_coord[2]-overlap_coord[0])*(overlap_coord[3]-overlap_coord[1])
    w_g = gt[2]-gt[0]
    h_g = gt[3]-gt[1]
    gt_area = w_g*h_g
    w_bbox = bbox[2]-bbox[0]
    h_bbox = bbox[3]-bbox[1]
    bbox_area = w_bbox*h_bbox
    union_area = bbox_area-overlap_area+gt_area
    if(union_area > 0):
        union_area = np.maximum(union_area,1e-10)
        iou = overlap_area/union_area
        iou = np.round(iou,4)
    else:
        iou=0
    return iou
        
def calc_pos(x_start,y_start,x_end,y_end,scalef,c):
    w = int((64)*pow(scalef,c))
    h = int((128)*pow(scalef,c))
    x_start = int(x_start*pow(scalef,c))
    y_start = int(y_start*pow(scalef,c))
    return x_start,y_start,w,h    

def detectMultiscale(original_img,scale,min_size):
    #inverse = 1/scale
    count=0
    positions = []
    #scores = []
    for img in pyramid(original_img,scale,min_size):
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        for (x,y) in sliding_window(img,step,wsize):
            x_start,y_start= x
            x_end,y_end = y
            
            patch = img[y_start:y_end,x_start:x_end]
            feat = hog(patch,(8,8),(2,2),12,'L2')[0]
            ans = clf2.predict([feat])
            if ans==1:
                score = clf2.decision_function([feat])
                x_start,y_start,w,h = calc_pos(x_start,y_start,x_end,y_end,scale,count)
                x_end = x_start+w
                y_end = y_start+h
                positions.append((int(x_start),int(y_start),int(x_end),int(y_end),score))
                #scores.append(score)        
        count+=1
        #print(count)
    return positions    
wsize = (128,64)
step = 48         
scale = 1.2
min_size = (128,64)
original_img = cv2.imread('/kaggle/input/inriaperson/Test/JPEGImages/crop001514.png')
positions_and_scores = detectMultiscale(original_img,scale,min_size)   
lam =  0.5
D = nms(positions_and_scoresc)
print(len(D))      
#print(len(positions_and_scores))
