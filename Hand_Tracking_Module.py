import cv2 as cv
import mediapipe as mp
import time

class HandDetection():
    def __init__(self,mode=False,maxHands=2,detectioncon=0.5,trackcon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectioncon=detectioncon
        self.trackcon=trackcon
        self.mphand=mp.solutions.hands
        self.mpdraw=mp.solutions.drawing_utils
        self.hand=self.mphand.Hands(self.mode,self.maxHands,self.detectioncon,self.trackcon)
    def findhands(self,img):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        results=self.hand.process(imgRGB)
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                self.mpdraw.draw_landmarks(img,handlms,self.mphand.HAND_CONNECTIONS)
                for id,lm in enumerate(handlms.landmark):
                    print(f'ID:{id} LandMarks:{lm}')
                    ih,iw,ic=img.shape
                    x,y=int(lm.x*iw),int(lm.y*ih)
                    print(f'Pixel values: ID:{id} X_value={x},y_Value={y}')
def main():
    ptime=0
    cap=cv.VideoCapture(0)
    detector=HandDetection()
    while True:
        success,img=cap.read()
        #print(success)
        detector.findhands(img)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
        cv.imshow('Image',img)
        if cv.waitKey(0) & 0XFF==ord('q'):
            break
