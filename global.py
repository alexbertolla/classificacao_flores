import os
import glob
import datetime
import tarfile
import urllib.request

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import mahotas
import numpy as np
import cv2
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







print('FIM')