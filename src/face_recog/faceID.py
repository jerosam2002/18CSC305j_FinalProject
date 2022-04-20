import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
import requests

images = []
classNames = []
encoded_face_train = []


def loadEncodings():
    print("[INFO] loading encodings...")
    data = pickle.loads(open("face_recog\encodings.pickle", "rb").read())
    for i in data["encodings"]:
        encoded_face_train.append(i)
    for i in data["names"]:
        classNames.append(i)
    #print(classNames)


def allow():
    print("[INFO] Starting Camera")
    breakLoop = False
    cap  = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    print("[INFO] Searching for Faces")
    while not breakLoop:
        success, img = cap.read()
        imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                y1,x2,y2,x1 = faceloc
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                cv2.destroyAllWindows()
                return True
        cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    loadEncodings()
    if(allow()):
        print('You are Authorized')

