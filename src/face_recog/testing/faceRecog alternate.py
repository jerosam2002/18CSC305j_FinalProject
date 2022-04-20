
import enum
from itertools import count
from tkinter.font import names
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

path = 'dataset/2'

images = []
classNames = []

"""mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)
"""
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())
encoded_face_train = data["encodings"]
for i in data["names"]:
    classNames.append(i)
print(classNames)
def sendData():
    pass

cap  = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    names = []
    for encode_face in encoded_faces:
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        if True in matches:
            matchedIdxs = [i for (i,b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name,0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        
        for((y1,x2,y2,x1),name) in zip(faces_in_frame,names):
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            sendData()
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break