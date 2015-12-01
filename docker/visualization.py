import cv2
import matplotlib.pyplot as plt
import numpy as np

import twinface


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    plt.imshow(out)
    plt.show()


def demonstrate_flann():
    img1 = cv2.imread('pairs/8/90477/90477d13.jpg')  # queryImage
    img2 = cv2.imread('pairs/8/90476/90476d17.jpg')  # trainImage

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    x, y, w, h = twinface.face_detect(img1_gray)
    face_image1 = img1_gray[y:y + h, x:x + w]

    x, y, w, h = twinface.face_detect(img2_gray)
    face_image2 = img2_gray[y:y + h, x:x + w]

    # Initiate SIFT detector
    sift = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(face_image1, None)
    kp2, des2 = sift.detectAndCompute(face_image2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda val: val.distance)

    # cv2.drawMatchesKnn expects list of lists as matches.
    drawMatches(face_image1, kp1, face_image2, kp2, matches[:200])

if __name__ == '__main__':
    demonstrate_flann()

