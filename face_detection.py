#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

start = time.perf_counter()

import os
import cv2
from cv2 import dnn_superres
from mtcnn import MTCNN

sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
path = "./models/EDSR_x4.pb" 
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 5)

detector = MTCNN()

cap = cv2.VideoCapture("../datasets/video 2.mp4")

#cap = cv2.VideoCapture(0)

filename = "face.jpg"

ret, image = cap.read()

os.chdir("./found/")

i = 0
while ret:
    #Capture frame-by-frame

    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if result:
        for person in result:
            bounding_box = person['box']
            print(person)

            x1 = bounding_box[0]
            y1 = bounding_box[1]
            x2 = x1 + bounding_box[2]
            y2 = y1 + bounding_box[3]

            frame = frame[y1:y2, x1:x2] # 
            x, y, __ = (frame.shape) 

            if x > 0 and y > 0:
              #display resulting frame
              #cv2.imshow('', frame)
              #cv2.waitKey(0)
          
              # Upscale the image
              result = sr.upsample(frame)

              # Save the image
              cv2.imwrite(f"face_{i}.jpg", result)

              i += 1

    ret, image = cap.read()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


end = time.perf_counter()

print("Time took = ", end - start)

#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()

