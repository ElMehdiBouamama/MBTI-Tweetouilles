#%% cell 0
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,128,255,cv2.THRESH_OTSU)

    edges = cv2.Canny(frame,100,100)

    cv2.imshow('Edges', th3)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
       break

cv2.destroyAllWindows()
cap.release()
