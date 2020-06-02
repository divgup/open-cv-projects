import hog
import numpy as np
from sklearn import preprocessing
#from sklearn.externals import joblib
import cv2
from sklearn import svm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#import helper
import matplotlib.pyplot as plt
MAX_HARD_NEGATIVES = 20000
import pickle
def crop_center(img):
    h,w = img.shape[:2]
    crop = []
    if w >=64 and h >=128:
        w = int((w-64)/2)
        h = int((h-128)/2)
        crop = img[h:h+128,w:w+64]
    return crop
def random_12_windows(img):
    h,w = img.shape[:2]
    w = w-64
    h = h-128
    #print(w,h)
    if w < 0 or h < 0:
        return []
    for i in range(12):
        x = np.random.randint(0,w)
        #if(h > 0)
        y = np.random.randint(0,h)
        cropimage = img[y:y+128,x:x+64]
        yield cropimage
            
method =  'L2'
feat = []
#pos = []
no_pos= 0
labels = []

for dirname, _, filenames in os.walk('/kaggle/input/inriaperson/Train/JPEGImages'):
    for filename in filenames:
        path_to_file = os.path.join(dirname, filename)
        extension = os.path.splitext(path_to_file)[1]
        if extension == ".png":
            #print(path_to_file)
            img = cv2.imread(path_to_file)
            img = cv2.resize(img,(96,160),interpolation=cv2.INTER_AREA)
            cropped = crop_center(img)
            cropped = cv2.normalize(cropped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #cropped = np.sqrt(cropped)
            #print(cropped)
            #plt.figure()
            #plt.imshow(cropped)
            #if no_pos > 2:
            #    break
            #resized = cv2.resize(img,(64,128))
            features = hog(cropped,(8,8),(2,2),12,method)[0]
            feat.append(features)
            labels.append(1)
            crop_flip = cv2.flip(cropped,1)
            features = hog(crop_flip,(8,8),(2,2),12,method)[0]
            feat.append(features)
            labels.append(1)
            no_pos+=1
        
#size = 400        
#neg = []
no_neg=0
for dirname, _, filenames in os.walk('/kaggle/input/zipdata/neg_train_images'):
    for filename in filenames:
        path_to_file = os.path.join(dirname, filename)
        extension = os.path.splitext(path_to_file)[1]
        if extension == ".jpg":
            img = cv2.imread(path_to_file)
            #img = cv2.resize(img,(96,160),interpolation=cv2.INTER_AREA)
            #cropped = crop_center(img)
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            for patch in random_12_windows(img):
                if patch.size==0:
                    continue
                features = hog(patch,(8,8),(2,2),12,method)[0]    
                feat.append(features)
                labels.append(0)
                no_neg+=1
print("positive images ",no_pos,"\t","negative images ",no_neg)            
# for dirname, _, filenames in os.walk('/kaggle/input/pedestrian-no-pedestrian/data/validation/no pedestrian'):      
clf = svm.LinearSVC(C=0.01,max_iter=1000,class_weight='balanced',verbose=1)
X_train,Y_train  = feat,labels
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train,Y_train = shuffle(X_train,Y_train,random_state=0)
clf.fit(X_train,Y_train)
print("trained")


Pkl_Filename = "linearModel1.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)


def hard_neg_mine():
    w_size = (128,64)
    step=8
    count1=0
    num = 0
    fp = []
    fp_labels= []
    for dirname, _, filenames in os.walk('/kaggle/input/zipdata/neg_train_images'):
        for filename in filenames:
            path_to_file = os.path.join(dirname, filename)
            extension = os.path.splitext(path_to_file)[1]
            if extension == ".jpg":
                img = cv2.imread(path_to_file)
                img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #print(img)
                for (x,y) in sliding_window(img,step,w_size):       
                    x_start,y_start= x
                    x_end,y_end = y
                    #print(x_start,x_end,y_start,y_end)
                    patch = img[y_start:y_end,x_start:x_end]
                    #print(patch)
                    #plt.imshow(patch)
                    feat = hog(patch,(8,8),(2,2),12,'L2')[0]
                    feat = np.array(feat)
                    ans = clf.predict([feat])
                    if ans==1:
                        #score = clf.decision_surface(feat)
                        fp.append(feat)
                        fp_labels.append(0)
                        count1+=1
                    if count1==MAX_HARD_NEGATIVES:
                        return np.array(fp),np.array(fp_labels)    
                print(" percentage hard neg mined",round((count1/MAX_HARD_NEGATIVES)*100)) 
            #sys.stdout.write("\r" + "\tHard Negatives Mined: " + str(count) + "\tCompleted: " + str(round((count / float(MAX_HARD_NEGATIVES))*100, 4)) + " %" )
            #sys.stdout.flush()
        num+=1    
    return np.array(fp),np.array(fp_labels) 
fp_feat,fp_labels = hard_neg_mine()
X_final = np.concatenate((X_train,fp_feat),axis=0)
Y_final = np.concatenate((Y_train,fp_labels),axis=0)
X_final,Y_final = shuffle(X_final,Y_final,random_state=0)
clf2 = svm.SVC(C=0.01,probability=True,gamma='auto',verbose=True)
clf2.fit(X_final,Y_final)
print("mining done")    
import pickle
# with open('/kaggle/input/object-detectmodel/Model2.pkl','rb') as pickle_file:
#     clf2= pickle.load(pickle_file)
# #clf = pickle.load(,encoding='bytes')
Pkl_Filename = "gaussModel2.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf2, file)
