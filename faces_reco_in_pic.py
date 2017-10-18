import face_recognition
from numpy.ma.core import array
from scipy.misc.common import face
import time
from os import walk




from funkce import *


duration_load_img = []          #casy nahravani obrazku
duration_encoding_img = []      #casy kodovani obrazku

filenames_unknow_faces = []           #nazvy souboru obrazku ktere nezname
filenames_know_faces = []

dir_unknow_faces = "./unknow_faces/"
dir_know_faces = "./know_faces/"



#do pole ulozi nazvy img soboru
#zname osoby
for (dirpath, dirnames, filenames) in walk(dir_know_faces):
    filenames_know_faces.extend(filenames)

#nezname osoby
for (dirpath, dirnames, filenames) in walk(dir_unknow_faces):
    filenames_unknow_faces.extend(filenames)

print("Pocet znamych osob: {} \nPocet neznamych osob: {}".format(len(filenames_know_faces), len(filenames_unknow_faces)))

'''
Import a encoding predelat, at si to automaticky nahrava ze slozky!!!
'''
# import img
print(">-- Nahranivani IMG a kodovani --<")
# delka nahravani img
print("NAHRAVAM IMG")

start_time = time.time()
bohuslav_sobotka_img = face_recognition.load_image_file("./know_faces/bohuslav_sobotka.jpg")
duration_load_img.append((time.time()-start_time)*1000)

start_time = time.time()
jaromir_jagr_img = face_recognition.load_image_file("./know_faces/jaromir_jagr.jpg")
duration_load_img.append((time.time()-start_time)*1000)

start_time = time.time()
leos_mares_img = face_recognition.load_image_file("./know_faces/leos_mares.jpg")
duration_load_img.append((time.time()-start_time)*1000)


#poznavani obrazku (ke kazdemu kodovani img vytvori pole o velikosti 128)
print("KODOVANI IMG")
start_time = time.time()
bohuslav_sobotka_encoding = face_recognition.face_encodings(bohuslav_sobotka_img)[0]
duration_encoding_img.append((time.time()-start_time)*1000)


start_time = time.time()
jaromir_jagr_encoding = face_recognition.face_encodings(jaromir_jagr_img)[0]
duration_encoding_img.append((time.time()-start_time)*1000)

start_time = time.time()
leos_mares_encoding = face_recognition.face_encodings(leos_mares_img)[0]
duration_encoding_img.append((time.time()-start_time)*1000)


#kazde vlozene pole obsahuje pole od velikosti 128
know_faces = [leos_mares_encoding,
              jaromir_jagr_encoding,
              bohuslav_sobotka_encoding]

for i, dur in enumerate(duration_load_img):
    print("\nCISLO {}".format(i+1))
    print("Delka nahravani: {} ms \nDelka kodovani: {} ms".format(dur, duration_encoding_img[i]))
    print("Velikost pole encoding: {} kB".format(get_size(know_faces[i])/1000))


for i, img in enumerate(filenames):
    print(img)
    print(dir_unknow_faces+img)

    unknow_image = face_recognition.load_image_file(dir_unknow_faces+img)

    # face_recognition.face_locations(image,model="hog") hog => CPU | cnn => GPU
    face_locations = face_recognition.face_locations(unknow_image)  # vrati souradnice v poli [(top, left, bottom, right),(...)]

    start_time = time.time()
    unknow_face_encoding = face_recognition.face_encodings(unknow_image)[0]
    results = face_recognition.compare_faces(know_faces, unknow_face_encoding)
    stop_time = time.time()

    duration = stop_time - start_time


    print("Pocet obliceju: {}".format(len(face_locations)))
    print("Souradnice obliceju: {}\n".format(face_locations))
    print("Cas vypoctu: {} ms".format(round(duration*1000,2)))


    print("Je to leos? {}".format(boolAnoNe(results[0])))
    print("Je to jaromir? {}".format(boolAnoNe(results[1])))
    print("Je to bohuslav? {}\n".format(boolAnoNe(results[2])))

'''
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
    print("Souradnice obliceju: {}\n".format(face_locations))
    print("Cas vypoctu: {} ms".format(int(round(duration*1000))))

    print("Je to leos? {}".format(boolAnoNe(results[0])))
    print("Je to jaromir? {}".format(boolAnoNe(results[1])))
    print("Je to bohuslav? {}\n".format(boolAnoNe(results[2])))
'''






