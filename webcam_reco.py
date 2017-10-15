'''

>> ! nezapomenout mit povolenou webcameru jinak vyhazuje chyby

- neni hotove: chybna prace s JPEG a nezobrazi nahled (TODO)

'''

import face_recognition
import cv2


video_capture = cv2.VideoCapture(0)

#nahrani zname osoby
me_know_image = face_recognition.load_image_file("./know_faces/me.jpg")
me_face_encoding = face_recognition.face_encodings(me_know_image)[0]

#pole
face_locations = []     #souradnice nalezenych obliceju
face_encodings = []     #detekovane tvare (rozpoznane)
face_names = []         #jmena obliceju

process_this_frame = True

while True:
    ret, frame = video_capture.read()  # detekuje kazdy frame

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)        #zmenci kazdy snimek

    if process_this_frame:
        face_locations = face_recognition.face_locations(small_frame)       #lokalizace obliceje na snimku
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([me_face_encoding], face_encoding)

            name = "NEZNAMY"

            if match[0]:
                name = "Michal"

            face_names.append(name)

    process_this_frame = not process_this_frame


    #Zobrazeni vysledku
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        #vrati zpet snimek z 1/4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)      #nakresli ctverec kolem tvare

        #nakresli plny ctverec (label) + jmeno
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)      #zobrazi vysledne obrazky

    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    '''

video_capture.release()
cv2.destroyAllWindows()