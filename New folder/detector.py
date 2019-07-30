import cv2, time

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt.xml')

video = cv2.VideoCapture(1)

a = 1

while True:

    a = a + 1

    check, frame = video.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.05, minNeighbors = 5)
    #print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for x,y,w,h in faces:
        img = cv2.rectangle(frame, (x,2), (x+w,y+h), (0,255,0), 3)
    cv2.imshow("Capture", img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    # print(check)
    # print(frame)
    #
    # time.sleep(3)
    #
    # cv2.imshow("Capture", frame)
    #
    # cv2.waitKey(0)


print(a)
video.release()
cv2.destroyAllWindows()

# face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt.xml')
#
# img = cv2.imread("barack-obama-12782369-1-402.jpg", 1)
#
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)
#
# print(type(faces))
# print(faces)
#
# for x,y,w,h in faces:
#     img = cv2.rectangle(img, (x,2), (x+w,y+h), (0,255,0), 3)
#
# resized = cv2.resize(img, (int(img.shape[1]*3), int(img.shape[0]*3)))
#
# cv2.imshow("Legend", resized)
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
