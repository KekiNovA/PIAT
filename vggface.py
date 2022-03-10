from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import os

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
  # load image from file
  pixels = pyplot.imread(filename)
  image = Image.fromarray(pixels)
  image = image.resize(required_size)
  face_array = asarray(image)
  return face_array

def my_extract_face(filename, required_size=(224, 224)):
  # load image from file
  pixels = pyplot.imread(filename)
  # create the detector, using default weights
  detector = MTCNN()
  # detect faces in the image
  results = detector.detect_faces(pixels)
  # extract the bounding box from the first face
  x1, y1, width, height = results[0]['box']

  x2, y2 = x1 + width, y1 + height
  # extract the face
  face = pixels[y1:y2, x1:x2]
  # resize pixels to the model size
  image = Image.fromarray(face)
  image = image.resize(required_size)
  face_array = asarray(image)
  return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
  # extract faces
  faces = []
  faces.append(my_extract_face(filenames[0]))
  faces += [extract_face(f) for f in filenames[1:]]
  samples = asarray(faces, 'float32')
  # prepare the face for the model, e.g. center pixels
  samples = preprocess_input(samples, version=2)
  # create a vggface model
  model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
  # perform prediction
  yhat = model.predict(samples)
  return yhat

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
  # calculate distance between embeddings
  score = cosine(known_embedding, candidate_embedding)
  if score <= thresh:
    print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
  else:
    print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


filenames = os.listdir("./found")
filenames = ["./found/" + file for file in filenames if ".DS" not in file] 
print(filenames)

# get embeddings file filenames
embeddings = get_embeddings(filenames)

# verify photos 
for i in range(1, len(filenames)):
  is_match(embeddings[0], embeddings[i])
