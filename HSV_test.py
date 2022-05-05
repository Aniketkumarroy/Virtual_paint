#! python3
import cv2 as cv
import numpy as np
imghsv=np.zeros((400,400,3),np.uint8)
cv.namedWindow("trackbar")
cv.resizeWindow("trackbar",640,240)
def empty(x):
    pass
cv.createTrackbar("hue","trackbar",0,179,empty)
cv.createTrackbar("sat","trackbar",0,255,empty)
cv.createTrackbar("val","trackbar",0,255,empty)
while True:
    hue=cv.getTrackbarPos("hue","trackbar")
    sat=cv.getTrackbarPos("sat","trackbar")
    val=cv.getTrackbarPos("val","trackbar")
    imghsv[:,:,0],imghsv[:,:,1],imghsv[:,:,2]=hue,sat,val
    img=cv.cvtColor(imghsv,cv.COLOR_HSV2BGR)
    cv.imshow("HSV",imghsv)
    cv.imshow("BGR",img)
    print(img)
    if cv.waitKey(1)&0xFF==ord('q'):
        break
cv.destroyAllWindows()