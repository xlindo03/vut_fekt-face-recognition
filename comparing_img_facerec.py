from os import walk
import os
import sys
import csv
import time
import itertools

import face_recognition

dir_list = []

with open("./img/csv.csv","r") as my_csv:
    csvReader = csv.reader(my_csv, delimiter='\n')
    raw_list = []
    for row in csvReader:
        #print(', '.join(row))
        raw_list.extend(row)

for i,x in enumerate(raw_list):
    if i > 0:
        dir_list.append(x)



for dir in dir_list:
    #print(dir)
    filename = []

    for (dirpath, dirnames, filenames) in walk(dir):
        filename.extend(filenames)
    filename.sort()

    #vztvoreni slozek pokud nejsou
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

        img_load = face_recognition.load_image_file(full_path)

        img_encode = face_recognition.face_encodings(img_load)[0]

        if img_encode.size != 128:
            print("-----# Nedetekovany oblicej: {}".format(full_path))
            face_not_detect += 1
        else:
            list_name.append(name)




    for (img1, img2) in itertools.combinations(list_name, 2):
        face_total_counter += 1


        img1_load = face_recognition.load_image_file(dir + "/" + img1)
        img1_encoding = face_recognition.face_encodings(img1_load)[0]

        img_encodings = []
        img_encodings.append(img1_encoding)

        img2_load = face_recognition.load_image_file(dir + "/" + img2)
        img2_encoding = face_recognition.face_encodings(img2_load)[0]


        results = face_recognition.face_distance(img_encodings, img2_encoding)
        print("{}|{}:{}".format(img1, img2,results[0]))

        with open("./img/data/" + dir + "/combination.csv", "a", newline='') as face_encoding_csv:
            face_encoding_Writer = csv.writer(face_encoding_csv)
            face_encoding_Writer.writerow(["{}|{}:{}".format(img1, img2,results[0])])

        with open("./img/data/combinations.csv", "a", newline='') as file_csv:
            csv_Writer = csv.writer(file_csv)
            csv_Writer.writerow(["{}:{}:{}|{}:{}".format(face_total_counter, dir, img1, img2, results[0])])


        if results <= 0.6:
            match_counter += 1
            # print(match_counter)

            if img1[:-7] != img2[:-7]:
                # print("Chyba")
                match_counter -= 1
                false_match_couter += 1


        if results > 0.6 and img1[:-7] == img2[:-7]:
            false_match_couter +=1


    print("TOTAL: {}".format(total_counter))
    print("TOTAL IMG: {}".format(face_total_counter))
    print("MATCH: {}".format(match_counter))
    print("FALSE: {}".format(false_match_couter))
    print("FACE NOT DETECT: {}".format(face_not_detect))

    with open("./img/data/combination_results.csv", "a", newline='') as csv_file:
        csv_Writer = csv.writer(csv_file)
        csv_Writer.writerow(["{}:{}:{}:{}:{}:{}".format(dir, total_counter,face_total_counter, match_counter, false_match_couter, face_not_detect)])

