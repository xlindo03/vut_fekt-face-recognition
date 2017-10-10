import face_recognition
from scipy.misc.common import face

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
    print("Obrazek číslo: ",i)
    unknow_image = face_recognition.load_image_file("./unknow_faces/{numb}.jpg".format(numb=i))
    unknow_face_encoding = face_recognition.face_encodings(unknow_image)[0]
    results = face_recognition.compare_faces(know_faces, unknow_face_encoding)

    print("Je to leos? {}".format(results[0]))
    print("Je to jaromir? {}".format(results[1]))
    print("Je to bohuslav? {}\n".format(results[2]))


