pos_img_dir = "/kaggle/input/inriaperson/Test/JPEGImages"
neg_img_dir = "/kaggle/input/pedestrian-no-pedestrian/data/validation/no pedestrian"
def read_images(f_pos, f_neg):

    print ("Reading Images")

    array_pos_features = []
    array_neg_features = []
    global total_pos_samples
    global total_neg_samples
    total_pos_samples = 0
    total_neg_samples=0
    for imgfile in f_pos:
        img = cv2.imread(os.path.join(pos_img_dir, imgfile))
        img = cv2.resize(img,(96,160),interpolation=cv2.INTER_AREA)
        #cropped = crop_center(img)
        cropped = crop_center(img)
        if len(cropped)==0:
            continue
        cropped = cv2.normalize(cropped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(cropped,(8, 8),(2, 2),12,"L2")[0]
        array_pos_features.append(features.tolist())

        total_pos_samples += 1

    for imgfile in f_neg:
        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        img = cv2.resize(img,(96,160),interpolation=cv2.INTER_AREA)
        cropped = crop_center(img)
        if len(cropped)==0:
            continue
        cropped = cv2.normalize(cropped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(cropped,(8, 8),(2, 2),12,"L2")[0]
        array_neg_features.append(features.tolist())
        total_neg_samples += 1
    return array_pos_features, array_neg_features    
def read_filenames():

    f_pos = []
    f_neg = []

    for (dirpath, dirnames, filenames) in os.walk(pos_img_dir):
        f_pos.extend(filenames)
        break

    for (dirpath, dirnames, filenames) in os.walk(neg_img_dir):
        f_neg.extend(filenames)
        break

    print("Positive Image Samples: " + str(len(f_pos)))
    print("Negative Image Samples: " + str(len(f_neg)))

    return f_pos, f_neg
pos_img_files, neg_img_files = read_filenames()
pos_features, neg_features = read_images(pos_img_files, neg_img_files)

pos_result = clf2.predict(pos_features)
neg_result = clf2.predict(neg_features)

true_positives = cv2.countNonZero(pos_result)
false_negatives = pos_result.shape[0] - true_positives

false_positives = cv2.countNonZero(neg_result)
true_negatives = neg_result.shape[0] - false_positives

print ("True Positives: " + str(true_positives), "False Positives: " + str(false_positives))
print ("True Negatives: " + str(true_negatives), "False Negatives: " + str(false_negatives))

precision = float(true_positives) / (true_positives + false_positives)
recall = float(true_positives) / (true_positives + false_negatives)

f1 = 2*precision*recall / (precision + recall)

print("Precision: " + str(precision), "Recall: " + str(recall))
print("F1 Score: " + str(f1))