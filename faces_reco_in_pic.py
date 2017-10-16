import face_recognition
from scipy.misc.common import face
import time

from funkce import *




#import img
bohuslav_sobotka_img = face_recognition.load_image_file("./know_faces/bohuslav_sobotka.jpg")
jaromir_jagr_img = face_recognition.load_image_file("./know_faces/jaromir_jagr.jpg")
leos_mares_img = face_recognition.load_image_file("./know_faces/leos_mares.jpg")





#poznavani obrazku
bohuslav_sobotka_encoding = face_recognition.face_encodings(bohuslav_sobotka_img)[0]
jaromir_jagr_encoding = face_recognition.face_encodings(jaromir_jagr_img)[0]
leos_mares_encoding = face_recognition.face_encodings(leos_mares_img)[0]



know_faces = [leos_mares_encoding,
              jaromir_jagr_encoding,
              bohuslav_sobotka_encoding]



for i in range(1,6):
    print("Obrazek cislo: ",i)



    unknow_image = face_recognition.load_image_file("./unknow_faces/{}.jpg".format(i))

    #face_recognition.face_locations(image,model="hog") hog => CPU | cnn => GPU
    face_locations = face_recognition.face_locations(unknow_image) # vrati souradnice v poli [(top, left, bottom, right),(...)]

    start_time = time.time()
    unknow_face_encoding = face_recognition.face_encodings(unknow_image)[0]
    results = face_recognition.compare_faces(know_faces, unknow_face_encoding)
    stop_time=time.time()

    duration=stop_time-start_time
    print(int(round(duration*1000.0)),"ms")

    print("Pocet obliceju: {}".format(len(face_locations)))


    print("Je to leos? {}".format(boolAnoNe(results[0])))
    print("Je to jaromir? {}".format(boolAnoNe(results[1])))
    print("Je to bohuslav? {}\n".format(boolAnoNe(results[2])))







