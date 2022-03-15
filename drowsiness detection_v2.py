import cv2
import os
from tensorflow.python.keras.models import load_model
import numpy as np
from time import sleep
import threading
import RPi.GPIO as GPIO
#Disable warnings (optional)
GPIO.setwarnings(False)
#Select GPIO mode
GPIO.setmode(GPIO.BCM)
#Set buzzer - pin 2 as output
#Set IR sensor - pin 5 as input
buzzer=2
light=5
lightInp=0
GPIO.setup(buzzer,GPIO.OUT)
GPIO.output(buzzer,GPIO.HIGH)
GPIO.setup(light,GPIO.IN)
beeperActive=False
#Function to activate beeping
def beeping(name):
    while beeperActive:
        GPIO.output(buzzer,GPIO.LOW)
        sleep(0.5)
        GPIO.output(buzzer,GPIO.HIGH)
        sleep(0.5)
face = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_righteye_2splits.xml')

threshold = 4

model = load_model('model/eye_model.h5')
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
rpred=[99]
lpred=[99]
msg = ['Driver Alert','Driver Drowsy','Seatbelt Fastened','Please Fasten Seatbelt']

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_ROI=frame
    gray_ROI=gray
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,255) if score>threshold or lightInp else (100,100,100) , 1 )
        frame_ROI=frame[y:y+h,x:x+w]
        gray_ROI=gray[y:y+h,x:x+w]
        break
    left_eye = leye.detectMultiScale(gray_ROI)
    right_eye =  reye.detectMultiScale(gray_ROI)

    cv2.rectangle(frame, (0,height-50) , (width,height) , (0,0,0) , thickness=cv2.FILLED )

    lightInp = GPIO.input(light)


    for (x,y,w,h) in right_eye:
        r_eye=frame_ROI[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        predict_r=model.predict(r_eye)
        rpred=np.argmax(predict_r,axis=1)
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame_ROI[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        predict_l=model.predict(l_eye)
        lpred=np.argmax(predict_l,axis=1)
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-2
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    if(score<0):
        score=0
    cv2.putText(frame,msg[1] if score>=threshold else msg[0],(100,height-20), font, 1,(0,0,255) if score>threshold else (0,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,msg[3] if lightInp else msg[2],(300,height-20), font, 1,(0,0,255) if lightInp else (0,255,0),1,cv2.LINE_AA)
    if(score>=threshold or lightInp) :
        if(not beeperActive):
            buzzThread = threading.Thread(target=beeping, args=(1,), daemon=True)
            beeperActive=True
            buzzThread.start()
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),4)    
    else:
        beeperActive=False
    cv2.imshow('Demo Output',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
