import datetime
import speech_recognition as sr
import pyttsx3
import sys
import geocoder
from gtts import gTTS
from playsound import playsound
import cv2
import numpy as np
import time
import random
import os
import keyboard
import yaml
from train import train
import finalrec as fr

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_eye.xml')

cap = cv2.VideoCapture(1)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')


# # Save
# name = {0:"Kartikeya", 1:"Srihamsini", 2:"Praveen", 3: "Lazo"}
# np.save('peopleNames.npy', name)
#
#
# # Load
# read_dictionary = np.load('peopleNames.npy', allow_pickle=True).item()
# print(read_dictionary)

now = datetime.date.today()
r = sr.Recognizer()

def startUp():
    print("Say something | Edith " + str(datetime.date.today()))
    message="Say any command, I'm listening"
    def my_speak(message):
        engine= pyttsx3.init()
        engine.say('{}'.format(message))
        engine.runAndWait()
    my_speak(message)

def restartProgram():
    print("Rebooting...")
    message="Rebooting"
    def my_speak(message):
        engine= pyttsx3.init()
        engine.say('{}'.format(message))
        engine.runAndWait()
    my_speak(message)
    python = sys.executable
    os.execl(python, python, * sys.argv)


def listen():
    try:
        with sr.Microphone(device_index=3) as source:
            audio = r.listen(source)
            text = r.recognize_google(audio, language='en-US')
            print("You said : {}".format(text))

            if format(text) in ('hi Edith', "sup", 'hi', "what's up"):
                hi()
                restartUp()

            elif format(text) in ('what is the date today', 'what day is today', 'what is the date', "what's today's date", "date today"):
                date()
                restartUp()

            elif format(text) in ('location', 'where am i', 'what is my location', 'give me the location', 'my location'):
                location()
                restartUp()

            elif format(text) in ('webcam', 'open the camera', 'open camera', 'camera'):
                webcamBoot()
                restartUp()

            elif format(text) in ('train', 'train the data', 'retrain data'):
                train()
                restartUp()

            elif format(text) in ('reboot', 'restart'):
                restartProgram()

            elif format(text) in ('add new face'):
                addNewFace()
                restartUp()

            elif format(text) in ('exit', 'shut down'):
                exit()

            else:
                print("Sorry, I did not recognize an authorized command, please try again")
                message="Sorry, I did not recognize an authorized command, please try again"
                def my_speak(message):
                    engine= pyttsx3.init()
                    engine.say('{}'.format(message))
                    engine.runAndWait()
                my_speak(message)
                restartUp()

    except Exception as e:
        if (str(e)==''):
            print("Sorry I could not recognize what you said.")
            message="Sorry I could not recognize what you said. I got no error message"
            def my_speak(message):
                engine= pyttsx3.init()
                engine.say('{}'.format(message))
                engine.runAndWait()

            my_speak(message)
            restartUp()
        else:
            print("Sorry I could not recognize what you said. The error was " + str(e))
            message="Sorry I could not recognize what you said. I got an error message."
            def my_speak(message):
                engine= pyttsx3.init()
                engine.say('{}'.format(message))
                engine.runAndWait()

            my_speak(message)
            restartUp()

    except IOError as er:
        print("Sorry I could not recognize what you said. The error was " + str(er))

def addNewFace():
    lst = list(os.listdir("./trainingimages"))
    read_dictionary = np.load('peopleNames.npy', allow_pickle=True).item()
    speak("What is the name of this new person?")
    personName = input("What is the name of this new person?")
    read_dictionary[int(max(lst)) + 1] = personName
    os.mkdir("./trainingimages/" + str(int(max(lst)) + 1))
    print("Booting webcam to add new face...")
    speak("Booting webcam to add new face")
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        cv2.imshow('img', img)
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            speak("Are you ready to start the facial setup procedure?")
            cv2.imshow('img', img)
            with sr.Microphone(device_index=3) as source:
                audio = r.listen(source)
                text = r.recognize_google(audio, language='en-US')
                if format(text) == 'yes':
                    speak("The facial setup procedure is about to commence, please follow the instructions as directed")
                    speak("Make sure your face is towards the center of the frame")
                    speak("Look at your screen at all times")
                    speak("I will start taking pictures in 5")
                    speak("4")
                    speak("3")
                    speak("2")
                    speak("1")
                    x = 1
                    while x < 11:
                        takePic(x,y,w,h, str(int(max(lst)) + 1))
                        x += 1

                    speak("Rotate your head 10 approximately 10 degrees to the left, make sure the blue square still recognizes your face")
                    speak("3")
                    speak("2")
                    speak("1")

                    x = 1
                    while x < 11:
                        takePic(x,y,w,h, str(int(max(lst)) + 1))
                        x += 1

                    speak("Now Rotate your head 10 approximately 10 degrees to the right, make sure the blue square still recognizes your face")
                    speak("3")
                    speak("2")
                    speak("1")

                    x = 1
                    while x < 11:
                        takePic(x,y,w,h, str(int(max(lst)) + 1))
                        x += 1

                    speak("Now Tilt your head 10 approximately 10 degrees up, make sure the blue square still recognizes your face")
                    speak("3")
                    speak("2")
                    speak("1")

                    x = 1
                    while x < 11:
                        takePic(x,y,w,h, str(int(max(lst)) + 1))
                        x += 1

                    speak("Now tilt your head 10 approximately 10 degrees down, make sure the blue square still recognizes your face")
                    speak("3")
                    speak("2")
                    speak("1")

                    x = 1
                    while x < 11:
                        takePic(x,y,w,h, str(int(max(lst)) + 1))
                        x += 1

                    speak("Now proceed to make random facial expressions as I proceed a burst picture sequence")
                    speak("3")
                    speak("2")
                    speak("1")

                    x = 1
                    while x < 11:
                        takePic(x,y,w,h, str(int(max(lst)) + 1))
                        x += 1

                    speak("The facial setup procedure is now complete, thank you for setting up your face")



def takePic(x,y,w,h, directory):
        randomNumber = random.randint(0,100)
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi_color = img[y-90:y+h+90, x-160:x+w+160]
        img_item = "./trainingimages/" + directory + "/" + str(randomNumber) + ".jpg"
        cv2.imwrite(img_item, roi_color)

def speak(message):
    engine= pyttsx3.init()
    engine.say('{}'.format(message))
    engine.runAndWait()

def webcamBoot():
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
            read_dictionary = np.load('peopleNames.npy', allow_pickle=True).item()
            predicted_name = str(read_dictionary[label])
            print(read_dictionary[label])
            fr.put_text(img, predicted_name, x, y)

            if keyboard.is_pressed('p'):
                takePic(x,y,w,h)


            eyes = eye_cascade.detectMultiScale(roi_gray)
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            if(confidence>120):
                dirname = ''
                speak("I've recognized a new face. Is this true? Yes or no?")
                newFace = input("I've recognized a new face. Is this true 'y' or 'n': ")
                try:
                    with sr.Microphone(device_index=3) as source:
                        audio = r.listen(source)
                        text = r.recognize_google(audio, language='en-US')
                        print("You said : {}".format(text))
                except Exception as e:
                    speak("Error recognizing what you said")
                    continue
                if(newFace == 'y') or (format(text) == 'yes'):
                    newDir = input("Would you like to create a new database for this face? 'y' or 'n': ")
                    if(newDir == 'y'):
                        lst = list(os.listdir("./trainingimages"))
                        print(lst)
                        personName = input("What is the name of this person?: ")
                        print("Folder ID: " + max(lst))
                        print(read_dictionary)
                        os.mkdir("./trainingimages/" + str(int(max(lst)) + 1))
                        #Add name to file
                        read_dictionary = np.load('peopleNames.npy', allow_pickle=True).item()
                        read_dictionary[int(max(lst)) + 1] = personName
                        takePic(x,y,w,h, str(int(max(lst)) + 1))
                        print("Directory successfully created")
                    else:
                        continue

                else:
                    continue
            else:
                continue

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break



    cap.release()
    cv2.destroyAllWindows()

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('./Cascades/data/haarcascade_frontalface_alt2.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

    return faces, gray_img

def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue


            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print("img_path", img_path)
            print("id: ", id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect, gray_img = faceDetection(test_img)
            if len(faces_rect)!=1:
                continue

            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y+w, x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))

    return faces, faceID

def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=1)

def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2)


def restartUp():
    print("Say any command")
    message="Say any command, I'm listening"
    def my_speak(message):
        engine= pyttsx3.init()
        engine.say('{}'.format(message))
        engine.runAndWait()
    my_speak(message)
    listen()

def date():
    message='Today is' + str(now)
    def my_speak(message):
        engine= pyttsx3.init()
        engine.say('{}'.format(message))
        engine.runAndWait()
    print('Today is ' + str(now))
    my_speak(message)

def hi():
    speech = gTTS(text="Hello there, Kartik", lang='en')
    speech.save('hi.mp3')
    playsound('hi.mp3')

def location():
    g = geocoder.ip('me')
    message= 'Your GPS coordinates are ' + str(g.latlng) + '. You are in ' + str(g.city)  + ', ' + str(g.state)
    def my_speak(message):
        engine= pyttsx3.init()
        engine.say('{}'.format(message))
        engine.runAndWait()
    print('Your GPS coordinates are ' + str(g.latlng) + '. You are in ' + str(g.city) + ', ' + str(g.state))
    my_speak(message)

def exit():
    message= 'Shutting Down'
    def my_speak(message):
        engine= pyttsx3.init()
        engine.say('{}'.format(message))
        engine.runAndWait()
    print('Shutting Down')
    my_speak(message)
