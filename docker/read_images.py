import pybrain
import cv2
import numpy as np
from scipy import misc
import os
import matplotlib.pyplot as plt
import pylab

NUM_FEATURES = 500


def read_file(filename):
    return misc.imread(filename)


def feature_extract(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_detect(img_gray)
    face_image = img[y:y+h, x:x+w]

    # sift = cv2.SIFT()
    # kp, des = sift.detectAndCompute(face_image, None)
    #
    # img2 = cv2.drawKeypoints(face_image, kp)
    # # plt.imshow(img2)
    # # plt.show()

    surf = cv2.SURF()
    surf.hessianThreshold = 500
    kp, des = surf.detectAndCompute(face_image, None)
    #
    # img2 = cv2.drawKeypoints(face_image, kp, None, (255,0,0), 4)
    # plt.imshow(img2)
    # plt.show()

    # fast = cv2.FastFeatureDetector()
    #
    # kp=fast.detect(face_image, None)
    # img2 = cv2.drawKeypoints(face_image, kp, None, (255, 0, 0))
    #
    # plt.imshow(img2)
    # plt.show()

    return kp, des


def face_detect(img_gray):
    face_cascade = cv2.CascadeClassifier('face-classifiers/haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(img_gray, minSize=(600, 600), scaleFactor=1.1, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    return face[0][0], face[0][1], face[0][2], face[0][3]


def build_dataset():
    DIR = 'tmp/'
    data = {}
    for x in os.listdir(DIR):
        data[x] = {}
        for f in os.listdir(DIR+x):
            data[x][f] = []
            for i in os.listdir(DIR+x+'/'+f):
                img = cv2.imread(DIR+x+'/'+f+'/'+i)
                kp, des = feature_extract(img)
                data[x][f].append((kp, des))
    return data


def calculate_feature_set(twin_set):
    labels = twin_set.keys()
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(twin_set[labels[0][1]], twin_set[labels[1][1]], k=2)

    ## for NUM_FEATURES in each image, possibly constrain the number of keypoints generated to NUM_FEATURES
    # make a feature for the keypoint
    # if it has a match, remove the matching keypoint from the other image's list.

    ## for each keypoint in each image
    # mark the feature 1 if it is present
    # else mark it 0


def construct_twin_classifier(twin_set):
    ds = pybrain.ClassificationDataSet(NUM_FEATURES, nb_classes=2, class_labels=twin_set.keys())
    for key, val in twin_set:
        for item in val:
            ds.appendLinked(item, [key])
    print ds.calculateStatistics()

def main():
    data = build_dataset()
    construct_twin_classifier(data[10])

if __name__ == '__main__':
    main()
