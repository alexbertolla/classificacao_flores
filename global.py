from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py


#--------------------
# tunable-parameters
#--------------------
images_per_class = 80
fixed_size       = tuple((500, 500))
train_path       = "dataset/train"
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'
bins             = 8

#HU MOMENTS DESCRIPTOR
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

#HALARICK DESCRIPTOR
def fd_halarick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    halarick = mahotas.features.haralick(gray).mean(axis=0)
    return halarick

#COLOR HISTOGRAM DESCRIPTOR
def color_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flattern()


#get the training labels
train_labels = os.listdir(train_path)

#sort the training labels
train_labels.sort()
print(train_labels)

#empty lists to hold feature vector and labels
global_features = []
labels = []

#loop over the training data sub-folders
for training_name in train_labels:
    #join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    #get the current training label
    current_label = training_name

    #loop over the imagens in each sub-folder
    for x in range(1, images_per_class+1):
        #get the image file name
        file = dir + "/" + str(x) + ".jpg"

        #read the image and resize it to a fixed-size
        image = cv2.imread(file)
        print(image)




print('FIM')