import cv2
import os
import numpy as np
import finalrec as fr
import pyttsx3

def train():

    print("Training data...")
    message="Training data"
    def my_speak(message):
        engine= pyttsx3.init()
        engine.say('{}'.format(message))
        engine.runAndWait()
    my_speak(message)

    img = './Known-Faces/Kartik.jpg'
    test_img = cv2.imread('./Known-Faces/13.jpg')
    faces_detected, gray_img = fr.faceDetection(test_img)
    print("faces_detected: ", faces_detected)

    # for (x,y,w,h) in faces_detected:
    #     cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=5)
    #
    # resized_img=cv2.resize(test_img, (int(test_img.shape[1]*2), int(test_img.shape[0]*2)))
    # cv2.imshow("face_detection", resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    faces, faceID = fr.labels_for_training_data('./trainingimages')
    face_recognizer = fr.train_classifier(faces, faceID)
    face_recognizer.save('trainingData.yml')

    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # face_recognizer.read('trainingData.yml')

    name = {0:"Kartikeya", 1:"Srihamsini", 2:"Praveen", 3: "Lazo"}

    for faces in faces_detected:
        (x, y, w, h) = faces
        roi_gray = gray_img[y:y+h, x:x+h]
        label, confidence = face_recognizer.predict(roi_gray)
        print("confidence: ", confidence)
        print("label: ", label)
        fr.draw_rect(test_img, faces)
        predicted_name = name[label]
        if(confidence>60):
            continue
        fr.put_text(test_img, predicted_name, x, y)

    resized_img=cv2.resize(test_img, (int(test_img.shape[1]), int(test_img.shape[0])))
    cv2.imshow("face_detection", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
