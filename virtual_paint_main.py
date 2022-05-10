import cv2
import numpy as np
import mediapipe as mp
# mp = mediapipe.solutions.mediapipe.python


class Painter():
    
    mp_hand = mp.solutions.hands
    hands = mp_hand.Hands(max_num_hands=1,min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    def __init__(self, colour, hsv=None, hand = False):
        self.colour = colour
        if hsv != None:
            self.lower = hsv[:3]
            self.upper = hsv[3:]
        else:
            self.upper = None
            self.lower = None
        self.hand = hand
        self.pnt = (-1,-1)

    def refresh_screen(self, img):
        blank1 = np.zeros(img.shape,np.uint8)
        blank2 = np.ones(img.shape,np.uint8) * 255
        self.screen = (blank1,blank2)
        self.prev_pnt = (-1,-1)
        self.pnt = (-1,-1)

    def detect_colour(self, img, maxarea):
        if self.hand == False:
            imghsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            if self.upper != None and self.lower != None:
                self.mask = cv2.inRange(imghsv,np.array(self.lower),np.array(self.upper))
                conts,_=cv2.findContours(self.mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                if len(conts) != 0:
                    cnt = max(conts,key=cv2.contourArea)
                    if cv2.contourArea(cnt) > maxarea:
                        x,y,w,_ = cv2.boundingRect(cnt)
                        self.pnt = (x+w//2, y)
            else:
                print("ERROR: hsv values not provided")
                print("obj = Painter(colour, hsv=None, hand=None")
                print("None value-------------^")
                exit()
        else:
            imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgrgb)
            if self.results.multi_hand_landmarks:
                x_norm, y_norm = self.results.multi_hand_landmarks[0].landmark[8].x, self.results.multi_hand_landmarks[0].landmark[8].y
                h, w = img.shape[:-1]
                self.pnt = (int(x_norm*w), int(y_norm*h))

    def update_screen(self, draw = True, radius = 5, line = 8, erase = False):
        if draw:
            if self.pnt[0] != -1:
                if erase:
                    cv2.circle(self.screen[0],self.pnt,radius,(0,0,0),-1)
                    cv2.circle(self.screen[1],self.pnt,radius,(255,255,255),-1)
                else:
                    cv2.circle(self.screen[0],self.pnt,radius,self.colour,-1)
                    cv2.circle(self.screen[1],self.pnt,radius,(0,0,0),-1)
                if self.prev_pnt[0] != -1:
                    if erase:
                        cv2.line(self.screen[0],self.pnt,self.prev_pnt,(0,0,0),line)
                        cv2.line(self.screen[1],self.pnt,self.prev_pnt,(255,255,255),line)
                    else:
                        cv2.line(self.screen[0],self.pnt,self.prev_pnt,self.colour,line)
                        cv2.line(self.screen[1],self.pnt,self.prev_pnt,(0,0,0),line)
                self.prev_pnt = self.pnt

    def draw(self, img, radius = 5):
        img = cv2.bitwise_and(img,self.screen[1])
        img = cv2.add(img,self.screen[0])
        cv2.circle(img,self.pnt,radius,self.colour,-1)
        return img

    def draw_hand(self, img, connections = True):
        if self.hand:
            if self.results.multi_hand_landmarks:
                if connections:
                    self.mp_draw.draw_landmarks(img,self.results.multi_hand_landmarks[0],self.mp_hand.HAND_CONNECTIONS)
                else:
                    self.mp_draw.draw_landmarks(img,self.results.multi_hand_landmarks[0])
        else:
            print("hand parameter should be True for this to work")
            print("obj = Painter(colour, hsv=None, hand=False")
            print("False value------------------------^")
            exit()

def UI(hand_paint, radius, line, Erase, Draw):
    blank = np.zeros((200,500,3), np.uint8)
    d = 1 if Draw else 100/255
    e = 1 if Erase else 100/255
    cv2.rectangle(blank,(0,0),(250,100),(0,int(255*d),0),-1)
    cv2.rectangle(blank,(250,0),(500,100),(0,0,int(255*e)),-1)
    cv2.putText(blank,"Draw",(50,70),cv2.FONT_HERSHEY_PLAIN,4,(0,int(200*d),0),3)
    cv2.putText(blank,"Erase",(300,70),cv2.FONT_HERSHEY_PLAIN,4,(0,0,int(200*e)),3)
    cv2.putText(blank,f"Line: {line}",(10,140),cv2.FONT_HERSHEY_PLAIN,2,(255,153,0),2)
    cv2.putText(blank,f"Radius: {radius}",(10,190),cv2.FONT_HERSHEY_PLAIN,2,(255,153,0),2)
    cv2.rectangle(blank,(200,120),(450,190),(255,255,255),2)
    text = "Hand on" if hand_paint.hand else "Hand off"
    cv2.rectangle(blank,(200,120),(450,190),hand_paint.colour,-1)
    cv2.putText(blank,text,(220,170),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    return blank



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    red = Painter(colour=(0,0,255), hsv=[169,120,79,179,255,255])
    purple = Painter(colour=(255,0,196), hsv=[116,45,76,149,255,255])

    paints = [red, purple]

    radius = 5
    line = 8
    DRAW = True
    ERASE = False
    hand_id = 0

    ret,img = cap.read()
    for p in paints:
        p.refresh_screen(img)

    for id,p in enumerate(paints):
        if p.hand:
            hand_id = id
            while (id < len(paints)):
                paints[id].hand = False
                id = id+1
            break
    
    L = len(paints)

    while True:
        ret, img = cap.read()
        if ret:
            img = cv2.flip(img,1)
            imgnew = cv2.GaussianBlur(img,(5,5),1)
            imgnew = cv2.bilateralFilter(imgnew, 9, 75, 75)
            for p in paints:
                p.detect_colour(imgnew,100)
                p.update_screen(DRAW, radius, line, ERASE)
                imgnew = p.draw(imgnew, radius)
                if p.hand:
                    p.draw_hand(imgnew)
            GUI = UI(paints[hand_id], radius, line, ERASE, DRAW)
            cv2.imshow("Painted", imgnew)
            cv2.imshow("window", GUI)
            # cv2.imshow("original", img)
            key = cv2.waitKey(1) & 0xff
            if key == ord('r'):
                for p in paints:
                    p.refresh_screen(img)
            elif key == ord('b'):
                radius = radius+1
            elif key == ord('s'):
                radius = max(radius-1,0)
            elif key == ord('l'):
                line = line+1
            elif key == ord('a'):
                line = max(line-1,0)
            elif key == ord('e'):
                ERASE = not ERASE
            elif key == ord('d'):
                DRAW = not DRAW
                for p in paints:
                    p.prev_pnt = (-1,-1)
            elif key == ord('h'):
                paints[hand_id].hand = not paints[hand_id].hand
            elif key == ord('c'):
                paints[hand_id].hand = False
                hand_id = (hand_id+1)%L
                paints[hand_id].hand = True
            elif key == ord('q'):
                break
        else:
            print("failed to start camera")
            break
    cap.release()
    cv2.destroyAllWindows()