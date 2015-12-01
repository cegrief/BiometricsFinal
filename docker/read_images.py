import pybrain
import cv2
import numpy as np
from scipy import misc
import os
import matplotlib.pyplot as plt
import pylab

NUM_FEATURES = 150


def read_file(filename):
    return misc.imread(filename)


def feature_extract(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(img_gray, None)

    img2 = cv2.drawKeypoints(img_gray, kp)
    plt.imshow(img2)
    plt.show()

    face_detect(img_gray)
    return kp, des


def face_detect(img_gray):
    face_cascade = cv2.CascadeClassifier('face-classifiers/haarcascade_frontalface_default.xml')

    face = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    return face.x, face.y, face.w, face.h


def build_dataset():
    ds = pybrain.datasets.SupervisedDataSet(NUM_FEATURES, 1)
    for x in os.walk('pairs'):
        for f in os.listdir(x[0]):
            img = cv2.imread(f)
            features = feature_extract(img)
            ds.addSample(features, (x[0],))


def main():
    kp, dis = feature_extract(cv2.imread('pairs/1/90457d13.jpg'))
    print dis

if __name__ == '__main__':
    main()
