import face_recognition
from numpy.ma.core import array
from scipy.misc.common import face
import time
from os import walk
import csv
from funkce import *


import dlib

dir_list = []

dir3264 = "rgb/3264x2448/"
dir3264_list = []

dir1632 = "rgb/1632x1224/"
dir1632_list = []

dir816 = "rgb/816x612/"
dir816_list = []

dir408 = "rgb/408x306/"
dir408_list = []

dir204 = "rgb/204x153/"
dir204_list = []

dir102 = "rgb/102x76/"
dir102_list = []

arnold = "arnold_schwarzenegger_"
arnold_list = []

dwayne = "dwayne_johnson_"
dwayne_list = []

sylvester = "sylvester_stallone_"
sylvester_list = []

vin = "vin_diesel_"
vin_list = []

know_img1 = "./rgb/arnold_schwarzenegger.jpg"
know_img2 = "./rgb/dwayne_johnson.jpg"
know_img3 = "./rgb/sylvester_stallone.jpg"
know_img4 = "./rgb/vin_diesel.jpg"

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

#nacteni CSV souboru
with open("csv.csv","r") as my_csv:
    #csvWriter = csv.writer(my_csv,delimiter=',')
    csvReader = csv.reader(my_csv, delimiter='\n')
    raw_list = []
    for row in csvReader:
        #print(', '.join(row))
        raw_list.extend(row)

for i,x in enumerate(raw_list):
    if i > 0:
        dir_list.append(x)



start = time.time()
load_know_img1 = face_recognition.load_image_file(know_img1)
print("Load IMG1: {}".format(start-time.time()))

start = time.time()
load_know_img2 = face_recognition.load_image_file(know_img2)
print("Load IMG2: {}".format(start-time.time()))

start = time.time()
load_know_img3 = face_recognition.load_image_file(know_img3)
print("Load IMG3: {}".format(start-time.time()))

start = time.time()
load_know_img4 = face_recognition.load_image_file(know_img4)
print("Load IMG4: {}".format(start-time.time()))


start = time.time()
face_location_know_img1 = face_recognition.face_locations(load_know_img1)
print("Location IMG1: {}".format(start-time.time()))

start = time.time()
face_location_know_img2 = face_recognition.face_locations(load_know_img2)
print("Location IMG2: {}".format(start-time.time()))

start = time.time()
face_location_know_img3 = face_recognition.face_locations(load_know_img3)
print("Location IMG3: {}".format(start-time.time()))

start = time.time()
face_location_know_img4 = face_recognition.face_locations(load_know_img4)
print("Location IMG4: {}".format(start-time.time()))

start = time.time()
encoding_know_img1 = face_recognition.face_encodings(face_location_know_img1)
print("Encoding IMG1: {}".format(start-time.time()))

start = time.time()
encoding_know_img1 = face_recognition.face_encodings(face_location_know_img2)
print("Encoding IMG2: {}".format(start-time.time()))

start = time.time()
encoding_know_img1 = face_recognition.face_encodings(face_location_know_img3)
print("Encoding IMG3: {}".format(start-time.time()))

start = time.time()
encoding_know_img1 = face_recognition.face_encodings(face_location_know_img4)
print("Encoding IMG4: {}".format(start-time.time()))


#############

for dir in dir_list:
    print(dir)

#############

print("3264x2448")
for img in dir3264_list:
    full_path = dir3264+"/"+img
    print("Processing {}.".format(full_path))

    start = time.time()
    load = face_recognition.load_image_file(full_path)
    print(" + Face load: {} sec".format(start-time.time()))

    start = time.time()
    face_location = face_recognition.face_locations(load)
    print(" + Location IMG1: {}".format(start - time.time()))

    start = time.time()
    encoding = face_recognition.face_encodings(face_location)[0]
    print(" + Face encoding: {}".format(start-time.time()))


    start = time.time()
    distance = face_recognition.face_distance(encoding_know_img1, encoding)
    print(" + Distance time: {} sec".format(start-time.time()))
    print("  + Distance s IMG1 {:0.3f}\n".format(distance))  # img1 = arnold

    start = time.time()
    distance = face_recognition.face_distance(encoding_know_img1, encoding)
    print(" + Distance time: {} sec".format(start-time.time()))
    print("  + Distance s IMG2 {:0.3f}\n".format(distance))  # img1 = arnold

    start = time.time()
    distance = face_recognition.face_distance(encoding_know_img1, encoding)
    print(" + Distance time: {} sec".format(start-time.time()))
    print("  + Distance s IMG3 {:0.3f}\n".format(distance))  # img1 = arnold

    start = time.time()
    distance = face_recognition.face_distance(encoding_know_img1, encoding)
    print(" + Distance time: {} sec".format(start-time.time()))
    print("  + Distance s IMG4 {:0.3f}\n".format(distance))  # img1 = arnold

    print("######\n------\n######")



print("#########\n#########")


