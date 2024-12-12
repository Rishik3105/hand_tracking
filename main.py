import mediapipe as mp
import cv2 as cv
import time
cap=cv.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands(False)
mpDraw=mp.solutions.drawing_utils
line_spec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3)  # Green lines with thickness 3
circle_spec = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=5)  # Red landmarks with radius 5
ptime=0
ctime=0
while True:
    success,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            #mpDraw.draw_landmarks(img,handlms) #THIS IS USED TO SHOW ALL THE 21 LANDMARK POINTS ON THE HAND
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS,circle_spec,line_spec) # THIS IS USED TO JOIN ALL THE LANDMARKS
    ctime=time.time()
    fps=1/(ctime-ptime) # fps= FRAMES PER SECOUND
    ptime=ctime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv.imshow('Captured Image',img)
    cv.waitKey(0)
