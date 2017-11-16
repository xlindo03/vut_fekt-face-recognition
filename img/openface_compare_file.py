import time

start = time.time()

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

import openface

from os import walk


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

dir3264 = "size/3264x2448/"
dir3264_list = []

dir1632 = "size/1632x1224/"
dir1632_list = []

dir816 = "size/816x612/"
dir816_list = []

dir408 = "size/408x306/"
dir408_list = []

dir204 = "size/204x153/"
dir204_list = []

dir102 = "size/102x76/"
dir102_list = []

arnold = "arnold_schwarzenegger_"
arnold_list = []

dwayne = "dwayne_johnson_"
dwayne_list = []

sylvester = "sylvester_stallone_"
sylvester_list = []

vin = "vin_diesel_"
vin_list = []

know_img1 = "./size/arnold_schwarzenegger.jpg"
know_img2 = "./size/dwayne_johnson.jpg"
know_img3 = "./size/sylvester_stallone.jpg"
know_img4 = "./size/vin_diesel.jpg"

#dir3264
for (dirpath, dirnames, filenames) in walk(dir3264):
    dir3264_list.extend(filenames)

#dir1632
for (dirpath, dirnames, filenames) in walk(dir1632):
    dir1632_list.extend(filenames)

#dir816
for (dirpath, dirnames, filenames) in walk(dir816):
    dir816_list.extend(filenames)

#dir408
for (dirpath, dirnames, filenames) in walk(dir408):
    dir408_list.extend(filenames)

#dir204
for (dirpath, dirnames, filenames) in walk(dir204):
    dir204_list.extend(filenames)

#dir102
for (dirpath, dirnames, filenames) in walk(dir102):
    dir102_list.extend(filenames)




'''
parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Def/home/michal/Desktop/vut_fekt-face-recognition-master/unknow_faces/1.jpg
/home/michal/Desktop/vut_fekt-face-recognition-master/unknow_faces/2.JPGault image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()


if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))
'''
start = time.time()
align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96) # imgDim 96

'''
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))
'''

def getRep(imgPath):
    print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(96, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))

    print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)


    print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
    #print("Representation:")
    #print(rep)
    print("-----\n")


    return rep



'''
for (img1, img2) in itertools.combinations(imgs, 2):
    d = getRep(img1) - getRep(img2)
    print("Comparing {} with {}.".format(img1, img2))
    print("  + Squared l2 distance between representations: {:0.3f}".format(np.dot(d, d)))

    eucl = np.linalg.norm(getRep(img1)-getRep(img2))
    print("EUCLIDIAN: {}".format(eucl))
'''



rep_know_img1 = getRep(know_img1)
rep_know_img2 = getRep(know_img2)
rep_know_img3 = getRep(know_img3)
rep_know_img4 = getRep(know_img4)


print("3264x2448")
for img in dir3264_list:
    full_path = dir3264+"/"+img
    #print dir3264_list
    img_rep = getRep(full_path)

    start = time.time()
    d_img1 = rep_know_img1 - img_rep
    print("Recognition: {} sec".format(time.time()-start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img1, d_img1)))  # img1 = arnold

    start = time.time()
    d_img2 = rep_know_img2 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG2 {:0.3f}\n".format(np.dot(d_img2, d_img2))) #img1 = arnold

    start = time.time()
    d_img3 = rep_know_img3 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img3, d_img3)))  # img1 = arnold

    start = time.time()
    d_img4 = rep_know_img4 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img4, d_img4)))  # img1 = arnold

    print("######\n------\n######")



print("#########\n#########")

print("1632x1224")
for img in dir1632_list:
    full_path = dir1632+"/"+img
    #print dir3264_list
    img_rep = getRep(full_path)

    start = time.time()
    d_img1 = rep_know_img1 - img_rep
    print("Recognition: {} sec".format(time.time()-start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img1, d_img1)))  # img1 = arnold

    start = time.time()
    d_img2 = rep_know_img2 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG2 {:0.3f}\n".format(np.dot(d_img2, d_img2))) #img1 = arnold

    start = time.time()
    d_img3 = rep_know_img3 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img3, d_img3)))  # img1 = arnold

    start = time.time()
    d_img4 = rep_know_img4 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img4, d_img4)))  # img1 = arnold

print("#########\n#########")

print("816x612")
for img in dir816_list:
    full_path = dir816+img
    #print dir3264_list
    img_rep = getRep(full_path)

    start = time.time()
    d_img1 = rep_know_img1 - img_rep
    print("Recognition: {} sec".format(time.time()-start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img1, d_img1)))  # img1 = arnold

    start = time.time()
    d_img2 = rep_know_img2 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG2 {:0.3f}\n".format(np.dot(d_img2, d_img2))) #img1 = arnold

    start = time.time()
    d_img3 = rep_know_img3 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img3, d_img3)))  # img1 = arnold

    start = time.time()
    d_img4 = rep_know_img4 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img4, d_img4)))  # img1 = arnold


print("#########\n#########")

print("408x306")
for img in dir408_list:
    full_path = dir408+"/"+img
    #print dir3264_list
    img_rep = getRep(full_path)

    start = time.time()
    d_img1 = rep_know_img1 - img_rep
    print("Recognition: {} sec".format(time.time()-start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img1, d_img1)))  # img1 = arnold

    start = time.time()
    d_img2 = rep_know_img2 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG2 {:0.3f}\n".format(np.dot(d_img2, d_img2))) #img1 = arnold

    start = time.time()
    d_img3 = rep_know_img3 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img3, d_img3)))  # img1 = arnold

    start = time.time()
    d_img4 = rep_know_img4 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img4, d_img4)))  # img1 = arnold

print("#########\n#########")

print("102x76")
for img in dir102_list:
    full_path = dir102+"/"+img
    img_rep = getRep(full_path)

    start = time.time()
    d_img1 = rep_know_img1 - img_rep
    print("Recognition: {} sec".format(time.time()-start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img1, d_img1)))  # img1 = arnold

    start = time.time()
    d_img2 = rep_know_img2 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG2 {:0.3f}\n".format(np.dot(d_img2, d_img2))) #img1 = arnold

    start = time.time()
    d_img3 = rep_know_img3 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img3, d_img3)))  # img1 = arnold

    start = time.time()
    d_img4 = rep_know_img4 - img_rep
    print("Recognition: {} sec".format(time.time() - start))
    print("  + Podobnost s IMG1: {:0.3f}\n".format(np.dot(d_img4, d_img4)))  # img1 = arnold

