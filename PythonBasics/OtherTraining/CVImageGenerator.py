#%% cell 0
import sys
import random
import cv2
import numpy as np
import math

def nothing(x):
    pass

value = 0
im0 = cv2.imread("C:\Licence Plate Images\Image00.jpg")
im1 = cv2.imread("C:\Licence Plate Images\Image01.jpg")
im0 = cv2.resize(im0,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_LINEAR)
im1 = cv2.resize(im1,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_LINEAR)
im2 = cv2.add(im0,im1)
cv2.imshow("Image 1",im2)
#cv2.namedWindow("Result")
#cv2.createTrackbar("Pourcentage","Result",value,100,nothing)
#while(1):
#    im2 = cv2.addWeighted(im0,value/100,im1,value/100,0)
#    cv2.imshow("Result",im2)
#    k = cv2.waitKey(1) & 0xFF
#    if(k == 27):
#        break
#    value = cv2.getTrackbarPos("Pourcentage","Result")

cv2.waitKey()
cv2.destroyAllWindows()