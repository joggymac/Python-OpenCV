import numpy as np
import cv2
from imutils.video import VideoStream

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
frames = []

while True:

    ret,im = cap.read()
    frames.append(im)

    if cv2.waitKey(10 == 27):
        break

frames = array(frames)



gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(cap,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = cap[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# sidefaces = faceside_cascade.detectMultiScale(gray)

# for (sx, sy, sw, sh) in sidefaces:
 #   cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

cv2.imshow('video', im)
cv2.waitKey(0)
cv2.destroyAllWindows()