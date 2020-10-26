#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:00:33 2020

@author: Doaa
"""
#================================ Importing Libraries ========================
import numpy as np
import cv2
from keras.models import load_model

#============================== Real Time ============================== 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name        
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
model = load_model('model_1_Eye.h5')
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()    
  if ret == True:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.1, 6)
      
      print('Number of faces detected:', len(faces))


      frame_with_detections = np.copy(frame)
      for (x,y,w,h) in faces:
          cv2.rectangle(frame_with_detections, (x,y), (x+w,y+h), (0,255,0), 3)
          bgr_crop = frame_with_detections[y:y+h, x:x+w] 
          orig_shape_crop = bgr_crop.shape
          gray_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
          eyes = eye_cascade.detectMultiScale(gray_crop, 1.1, 6)
          if len(eyes) != 2:
              print("no eyes detected")
          resize_gray_crop = cv2.resize(gray_crop, (96, 96)) / 255.0
          
          landmarks = model.predict(resize_gray_crop.reshape(1,96,96,1))[0]
          d1= ((landmarks[0::2])* np.float32(orig_shape_crop[0])/96.0) + +np.float32(x)
          d2= ((landmarks[1::2])* np.float32(orig_shape_crop[1])/96.0) + +np.float32(y)
          cv2.circle(frame_with_detections, (d1[0],d1[1]), 1, (255,0,0), 3)
          cv2.circle(frame_with_detections, (d2[0],d2[1]), 1, (255,0,0), 3)

      # Display the resulting frame
      frame_with_detections = cv2.resize(frame_with_detections, (960, 540))
      cv2.imshow("Frame",frame_with_detections)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
          break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
