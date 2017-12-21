#%% cell 0
import cv2
import numpy as np
import matplotlib.pyplot as plt

#"C:\Licence Plate Images\Video01.mp4"
cap = cv2.VideoCapture(0)

blur = 1
morphology = 3
erosion = 8
dilatation = 13
seconderosion = 1
def onChange(x):
    pass
    
cv2.namedWindow("BluredImage")
cv2.createTrackbar("Blur","BluredImage",blur,20,onChange)
cv2.namedWindow("MorphologyEx")
cv2.createTrackbar("Morph", "MorphologyEx",morphology,20,onChange)
cv2.createTrackbar("Erosion", "MorphologyEx",erosion,20,onChange)
cv2.createTrackbar("Dilatation", "MorphologyEx",dilatation,20,onChange)
cv2.createTrackbar("SecondErosion", "MorphologyEx",seconderosion,20,onChange)
frame_counter = 0
while(True):
    ret,frame = cap.read()
    ResizedImg = cv2.resize(frame,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_CUBIC)
    GSImg = cv2.cvtColor(ResizedImg,cv2.COLOR_RGB2GRAY)
    BluredImg = cv2.GaussianBlur(GSImg,(blur*2+1,blur*2+1),0)
    dilatationkernel = np.ones((dilatation*2+1,dilatation*2+1),np.uint8)
    TresholdedImg = cv2.dilate(BluredImg,dilatationkernel)
    erosionkernel = np.ones((erosion*2+1,erosion*2+1),np.uint8)
    TresholdedImg = cv2.erode(BluredImg,erosionkernel)
    
    ret3,TresholdedImg = cv2.threshold(TresholdedImg,200,255,cv2.THRESH_OTSU)
    cv2.imshow("MorphologyEx",TresholdedImg)
    Contouredimg, contours, hierarchy = cv2.findContours(TresholdedImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    kernel = np.ones((morphology*2+1,morphology*2+1),np.uint8)
    Contouredimg = cv2.morphologyEx(Contouredimg,cv2.MORPH_GRADIENT,kernel)
    erosionkernel = np.ones((erosion*2+1,erosion*2+1),np.uint8)
    Contouredimg = cv2.erode(Contouredimg,erosionkernel)
    dilatationkernel = np.ones((dilatation*2+1,dilatation*2+1),np.uint8)
    Contouredimg = cv2.dilate(Contouredimg,dilatationkernel)
    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStatsWithAlgorithm(Contouredimg,8,cv2.CV_32S,cv2.CCL_DEFAULT)

    for label in range(1,labelnum):
        x,y = GoCs[label]
        img = cv2.circle(ResizedImg, (int(x),int(y)), 1, (0,0,255), -1)    
        
        x,y,w,h,size = contours[label]
        ratio = w/h
        #if(ratio <= 4 and ratio > 3 and (w >= 100 and h >=30) or (w >=30 and h >= 100) ):
        img = cv2.rectangle(ResizedImg, (x,y), (x+w,y+h), (255,255,0), 1)

    cv2.imshow("BluredImage",ResizedImg)
    blur = cv2.getTrackbarPos("Blur","BluredImage")
    morphology = cv2.getTrackbarPos("Morph", "MorphologyEx")
    erosion = cv2.getTrackbarPos("Erosion", "MorphologyEx")
    dilatation = cv2.getTrackbarPos("Dilatation", "MorphologyEx")
    #cv2.imshow("BluredImage", BluredImg)
    #cv2.imshow("MorphologyEx", Contouredimg)
    #cv2.imshow("FinalResult", ResizedImg)
    #if ret==True:
        #out.write(ResizedImg)
    frame_counter += 1
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0 #Or whatever as long as it is the same as next line
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()

