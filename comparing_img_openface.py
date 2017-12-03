from os import walk
import sys
import csv
import time
import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

import openface



dir = "img/"
dir_list = []

align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96) # imgDim 96

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')



def getRep(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Nelze nahrat img: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Nenalezeny oblicej: {}".format(imgPath))

    alignedFace = align.align(96, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Img nelze zarovnat: {}".format(imgPath))

    rep = net.forward(alignedFace)

    return rep




dir_list = []

with open("./img/csv.csv", "r") as my_csv:
    csvReader = csv.reader(my_csv, delimiter='\n')
    raw_list = []
    for row in csvReader:
        # print(', '.join(row))
        raw_list.extend(row)

for i, x in enumerate(raw_list):
    if i > 0:
        dir_list.append(x)

for dir in dir_list:
    # print(dir)
    filename = []

    for (dirpath, dirnames, filenames) in walk(dir):
        filename.extend(filenames)
    filename.sort()

    # vztvoreni slozek pokud nejsou
    if os.path.isdir("./img/data/" + dir) != True:
        os.makedirs("./img/data/" + dir)

    print("\n\n#######################\n>>>> {} <<<<\n#######################".format(dir))

    total_counter = 0
    match_counter = 0
    false_match_couter = 0
    face_not_detect = 0
    face_total_counter = 0

    img_encoding = []
    img_name = []

    list_name = []

    for name in filename:
        total_counter += 1
        full_path = dir + "/" + name
        print(full_path)

        img_encode = getRep(full_path)

        if img_encode.size != 128:
            print("-----# Nedetekovany oblicej: {}".format(full_path))
            face_not_detect += 1
        else:
            list_name.append(name)

    for (img1, img2) in itertools.combinations(list_name, 2):
        face_total_counter += 1

        d = getRep(img1) - getRep(img2)
        results = np.dot(d_img, d_img)

        print("{}|{}:{}".format(img1, img2, results))

        with open("./img/data/" + dir + "/combination.csv", "a", newline='') as face_encoding_csv:
            face_encoding_Writer = csv.writer(face_encoding_csv)
            face_encoding_Writer.writerow(["{}|{}:{}".format(img1, img2, results)])

        with open("./img/data/combinations.csv", "a", newline='') as file_csv:
            csv_Writer = csv.writer(file_csv)
            csv_Writer.writerow(["{}:{}:{}|{}:{}".format(face_total_counter, dir, img1, img2, results)])

        if results <= 1:
            match_counter += 1
            # print(match_counter)

            if img1[:-7] != img2[:-7]:
                # print("Chyba")
                match_counter -= 1
                false_match_couter += 1

        if results > 1 and img1[:-7] == img2[:-7]:
            false_match_couter += 1

    print("TOTAL: {}".format(total_counter))
    print("TOTAL FACE: {}".format(face_total_counter))
    print("MATCH: {}".format(match_counter))
    print("FALSE: {}".format(false_match_couter))
    print("FACE NOT DETECT: {}".format(face_not_detect))

    with open("./img/data/combination_results.csv", "a", newline='') as csv_file:
        csv_Writer = csv.writer(csv_file)
        csv_Writer.writerow(["{}:{}:{}:{}:{}:{}".format(dir, total_counter, face_total_counter, match_counter,
                                                        false_match_couter, face_not_detect)])



