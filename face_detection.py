#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#For MTCNN
import time
import os
import cv2
from cv2 import dnn_superres
from mtcnn import MTCNN

#For VGGFace
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# extract a single face from a given photograph
def extract_face(pixels, required_size=(224, 224)):
  # load image from file
  image = Image.fromarray(pixels)
  image = image.resize(required_size)
  face_array = asarray(image)
  return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(frame_dict):
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # extract faces
    faces = []
    for key in frame_dict:
        faces.append(extract_face(frame_dict[key]))

    images = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    images = preprocess_input(images, version=2)
    # perform prediction
    predicted_images = model.predict(images)
    for i in enumerate(frame_dict):
      frame_dict[i[1]] = predicted_images[i[0]]
        

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
  # calculate distance between embeddings
  score = cosine(known_embedding, candidate_embedding)
  if score <= thresh:
    return True
  else:
    return False


def start(video, image):

  start = time.time()

  #MTCNN and SuperResolution
  sr = dnn_superres.DnnSuperResImpl_create()

  # Read the desired model
  path = "./models/EDSR_x4.pb" 
  sr.readModel(path)


  # Set the desired model and scale to get correct pre- and post-processing
  sr.setModel("edsr", 5)

  detector = MTCNN()


  cap = cv2.VideoCapture(video)

  #cap = cv2.VideoCapture(0)


  #adding given image in dictionary
  frame_dict = dict()

  image = cv2.imread(image)
  result = detector.detect_faces(image)

  # extract the bounding box from the first face
  x1, y1, width, height = result[0]['box']
  x2, y2 = x1 + width, y1 + height
  # extract the face
  face = image[y1:y2, x1:x2]
  # resize pixels to the model size
  image = Image.fromarray(face)
  image = image.resize((224, 224))
  frame_dict["given_face"] = asarray(image)

  # Read video file
  ret, image = cap.read()
  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = 0

  while ret:
      #Capture frame-by-frame
      frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      #frame = cv2.resize(frame, (400, 400),fx=0,fy=0, interpolation=cv2.INTER_CUBIC)
      frame_count += 1
      
      ret, image = cap.read()
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

      if frame_count % 5 == 0:
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

                frame = frame[y1:y2, x1:x2] 
                x, y, __ = (frame.shape) 

                if x > 0 and y > 0:
                  # Upscale the image
                  print("Face size before superres: ", frame.shape)

                  result = sr.upsample(frame)
                  #result = frame

                  print("face size after superres", result.shape)
                  time_stamp = float(frame_count)/fps
                  frame_dict[time_stamp] = result


  print("Total frames: ", frame_count)

  #When everything's done, release capture
  cap.release()

  #VGGFace

  get_embeddings(frame_dict)

  count = 0
  for key in frame_dict:
      if key == "given_face":
          continue
      elif is_match(frame_dict["given_face"], frame_dict[key]) == True:
          print("Timestamp:", key)
          count += 1

  return frame_dict 
          
  print("Timestamps generated: ", count)

  end = time.time()

  print(f'It took {end - start} seconds!')
