# example of horizontal shift image augmentation
import cv2
import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# load the image
img = load_img("../datasets/Person 6_var1.jpg")

# convert to numpy array
data = img_to_array(img)

# expand dimension to one sample
samples = expand_dims(data, 0)

# create image data augmentation generator
datagen = ImageDataGenerator(
  rotation_range=45,
  width_shift_range=0.3,
  height_shift_range=0.3,
  horizontal_flip=True,
  fill_mode='nearest')

# Prepare iterator
it = datagen.flow(samples, batch_size=1)

os.chdir("./train")

for i in range(70):
  # generate batch of images
  batch = it.next()
  # convert to unsigned integers for viewing
  image = batch[0].astype('uint8')

  cv2.imwrite(f"person1_{i}.jpg", image)

os.chdir("../test")

for i in range(30):
  # generate batch of images
  batch = it.next()
  # convert to unsigned integers for viewing
  image = batch[0].astype('uint8')

  cv2.imwrite(f"person1_{i}.jpg", image)
