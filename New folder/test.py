import os
import cv2

list = list(os.listdir("./trainingimages"))
print(list)
face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')
files = folders = 0


for _, dirnames, filenames in os.walk("./trainingimages"):
  # ^ this idiom means "we won't be using this value"
    files += len(filenames)
    folders += len(dirnames)
    print(files)

files = []
fileNumber = -1

basepath = "./Training Images/Kartikeya Gullapalli"
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        files.append(entry)

    fileNumber += 1
    img = cv2.imread("./Training Images/Kartikeya Gullapalli/" + files[fileNumber])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("sup", img)
    print(files[fileNumber])


print(fileNumber)
print(folders)
print(list[1])

cv2.waitKey(0)
cv2.destroyAllWindows()
