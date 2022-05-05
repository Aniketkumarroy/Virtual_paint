#! python3
import cv2
import numpy as np
import time

#colours = [colour_data1, colour_data2,...]
#colour_data1 = [hue_min, hue_max, sat_min, sat_max, val_min, val_max, (colour1_b, colour1_g, colour1_r)]
#for more accurate results you can tune the values..............
colours = [[169,120,79,179,255,255,(0,0,255)],  #RED  
          [116,45,76,149,255,255,(255,0,196)]]  #PURPLE 

#screen = {(colour1_b, colour1_g, colour1_r):(img1,img2), ....}
screen = {}
#prev_points = {(colour1_b, colour1_g, colour1_r):(prev_x, prev_y), ....}
prev_points = {} #store the previous point of the different colours
ERASE = False #whether to erase
DRAW = True   #whether to draw
RADIUS = 2    #radius of tip
gui_size = 60 
prev = 0      #for calculating fps
def preprocess(img):
    img = cv2.bilateralFilter(img, 9, 75, 75)
    # img = cv2.flip(img,1)
    return img

def segment(img,Color):
    imghsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imghsv,np.array(Color[:3]),np.array(Color[3:6]))
    # cv2.imshow(f"{Color[0]}",mask)
    return mask

def contour(img):
    x,y,w,h = -1,-1,-1,-1
    conts,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(conts) != 0:
        cnt = max(conts,key=cv2.contourArea)
        if cv2.contourArea(cnt) > 100:
            x,y,w,h = cv2.boundingRect(cnt)
    return x,y,w,h

def draw(img,screen):
    for colour in screen.keys():
        img1, img2 = screen[colour]
        img = cv2.bitwise_and(img,img2)
        img = cv2.add(img1,img)
    return img

def refresh_screen(img):
    for i in colours:
        blank1 = np.zeros(img.shape,np.uint8)
        blank2 = np.ones(img.shape,np.uint8) * 255
        screen[i[6]] = (blank1,blank2)
        prev_points[i[6]] = (-1,-1)

def update_screen(img,screen,prev_points,RADIUS,DRAW):
    for col in colours:
        mask = segment(img,col[0:6])
        x,y,w,h = contour(mask)

        if x != -1 and DRAW: # if points is found and DRAW is True
            img1, img2 = screen[col[6]]
            prev_x, prev_y = prev_points[col[6]]
            if prev_x != -1: # if the found point is not the very first point
                if ERASE:    # to erase the paint
                    cv2.line(img1,(x+w//2,y),(prev_x,prev_y),(0,0,0),RADIUS)
                    cv2.line(img2,(x+w//2,y),(prev_x,prev_y),(255,255,255),RADIUS)
                else:        # draw
                    cv2.line(img1,(x+w//2,y),(prev_x,prev_y),col[6],RADIUS)
                    cv2.line(img2,(x+w//2,y),(prev_x,prev_y),(0,0,0),RADIUS)
            screen[col[6]] = (img1, img2)    # update screen
            prev_points[col[6]] = (x+w//2,y) # update Previous Points
            cv2.circle(img,(x+w//2,y),RADIUS,col[6],-1)
    return screen, prev_points

def GUI(img,h,w,DRAW,LINE):
    pass

cap = cv2.VideoCapture(0)
print("Starting Camera")
time.sleep(0.5)

ret, img = cap.read()
refresh_screen(img)

while True:
    ret, img = cap.read()
    img = cv2.flip(img,1)
    imgoutput = img.copy()
    if ret:
        img = preprocess(img)
        screen, prev_points = update_screen(imgoutput,screen,prev_points,RADIUS,DRAW)
        imgoutput = draw(imgoutput,screen)
        imgoutput = cv2.resize(imgoutput,None,fx=1.5,fy=1.5)
        # height, width = imgoutput.shape[:-1]
        # output = np.zeros((height+gui_size,width,3),np.uint8)
        # output[gui_size:gui_size+height,0:width,:] = imgoutput
        now = time.time()
        cv2.putText(img,f"{1//(now-prev)} fps",(20,50),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),2)
        cv2.imshow("original",img)
        cv2.imshow("paint",imgoutput)
        prev = now
        key = cv2.waitKey(1) & 0xff
        if key == ord('r'):
            refresh_screen(img)
        elif key == ord('b'):
            RADIUS = RADIUS+1
        elif key == ord('s'):
            RADIUS = max(RADIUS-1,0)
        elif key == ord('e'):
            ERASE = not ERASE
        elif key == ord('d'):
            DRAW = not DRAW
            for col in colours:
                prev_points[col[6]] = (-1,-1)
        elif key == ord('q'):
            break
    else:
        print("failed to start camera")
        break
cap.release()
cv2.destroyAllWindows()