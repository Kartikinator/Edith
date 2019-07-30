import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
import keyboard
import finalrec as fr

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_eye.xml')

cap = cv2.VideoCapture(1)

# faces, faceID = fr.labels_for_training_data('./trainingimages')
# face_recognizer = fr.train_classifier(faces, faceID)
# face_recognizer.save('trainingData.yml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')

name = {0:"Kartikeya", 1:"Srihamsini", 2:"Praveen", 3: "Lazo"}

def takePic():
    randomNumber = random.randint(0,10001)
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_color = img[y-90:y+h+90, x-160:x+w+160]
    img_item = "./trainingimages/4/" + str(randomNumber) + ".jpg"
    cv2.imwrite(img_item, roi_color)


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi_gray)
        print("confidence: ", confidence)
        print("label: ", label)
        predicted_name = name[label]
        if(confidence>60):
            continue
        fr.put_text(img, predicted_name, x, y)

        if keyboard.is_pressed('p'):
            takePic()


        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break



cap.release()
cv2.destroyAllWindows()
