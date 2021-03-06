import cv2
from scipy import misc
import os
import csv


def read_file(filename):
    return misc.imread(filename)


def feature_extract(img, t='combo'):
    # uses SIFT, SURF, ORB, or a combination of all three to extract features from the face
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_detect(img_gray)
    face_image = img[y:y+h, x:x+w]
    kp_out = []
    des_out = []

    if t == 'sift' or t == 'combo':
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(face_image, None)
        kp_out.append(kp)
        des_out.append(des)

    if t == 'surf' or t == 'combo':
        surf = cv2.SURF()
        surf.hessianThreshold = 500
        kp, des = surf.detectAndCompute(face_image, None)
        kp_out.append(kp)
        des_out.append(des)

    if t == 'combo':
        orb = cv2.ORB()
        kp, des = orb.detectAndCompute(face_image, None)
        kp_out.append(kp)
        des_out.append(des)

    return kp_out, des_out


def face_detect(img_gray):
    # uses Haar cascade to classify the face in our image
    face_cascade = cv2.CascadeClassifier('face-classifiers/haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(img_gray, minSize=(600, 600), scaleFactor=1.1, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    return face[0][0], face[0][1], face[0][2], face[0][3]


def build_dataset(t):
    # constructs the dataset that will be used for training
    DIR = 'tmp/'
    data = {}
    for x in os.listdir(DIR):
        data[x] = {}
        for f in os.listdir(DIR+x):
            data[x][f] = {'kp':[], 'des':[]}
            for i in os.listdir(DIR+x+'/'+f):
                img = cv2.imread(DIR+x+'/'+f+'/'+i)
                kp, des = feature_extract(img, t)
                data[x][f]['kp'].append(kp)
                data[x][f]['des'].append(des)
    return data


def find_nearest_neighbor(in_image, twin_set):
    # classifies an image as either one twin or the other by using FLANN or brute force classification techniques and
    # calculates the distances for each image
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    total_distances = []
    for key in twin_set.keys():
        # for each example for this class:
        for example in twin_set[key]['des']:
            # for each feature type
            dist = 0
            for i in xrange(len(example)):
                # find matches between input data and twin data
                # if orb, use the orb matcher
                if i == 2:
                    matches = bf_orb.match(in_image['des'][i], example[i])
                else:
                    matches = flann.match(in_image['des'][i], example[i])

                if len(matches) < 100:
                    dist += 1
                    continue
                distances = [m.distance for m in matches]
                # normalize the distances to a value between 0 and 1 and sum them.
                normalized_distances = [x/max(distances) for x in distances]
                dist += sum(normalized_distances)/len(matches)

                # penalize non-matched features
                dist += float(len(in_image['des'][i])-len([matches]))/len(in_image['des'][i])

            total_distances.append((key, dist/len(example)))

    sorted_distances = sorted(total_distances, key=lambda x: x[1])
    print sorted_distances
    return sorted_distances


def load_input(path, t):
    # reads in image so we can extract the features from it appropriately
    img = cv2.imread(path)
    kp, des = feature_extract(img, t)
    out = {}
    out['kp'] = kp
    out['des'] = des
    return out


def run_experiment(t):
    # performs our experiment, attempting to classify images based on the twin pair it belongs to
    testdir = 'test/'
    data = build_dataset(t)

    error = 0
    count = 0
    distances = []
    for twinpair in os.listdir(testdir):
        for pic in os.listdir(testdir+twinpair):
            inp = load_input(testdir+twinpair+'/'+pic, t)
            expected_class = pic[:5]
            classification = find_nearest_neighbor(inp, data[twinpair])
            distances.append([x[1] for x in classification])
            print "Expected: ", expected_class
            print "Received: ", classification[0][0]

            if expected_class != classification[0][0]:
                error += 1
            count += 1

    print "Error: ", float(error)/count
    return error, distances


def main():
    t = 'sift'
    error, distances = run_experiment(t)

    writer = csv.writer(open(t+'.csv', 'wb'))
    writer.writerows(distances)

if __name__ == '__main__':
    main()
