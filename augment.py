# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# load the image
img = load_img('./data/person0.jpg')

# convert to numpy array
data = img_to_array(img)

batch_size = 1

# expand dimension to one sample
samples = expand_dims(data, 0)

# create image data augmentation generator
datagen = ImageDataGenerator(width_shift_range=[-70,70])

# prepare iterator
it = datagen.flow(samples, batch_size=batch_size)

a = 0

# generate samples and plot
for i in range(20):

	# generate batch of images
	batch = it.next()

	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')

	# plot raw pixel data
	pyplot.imsave(f"train/person0/person0_{a}.jpg", image);a += 1 

# --- Flip

# create image data augmentation generator
datagen = ImageDataGenerator(horizontal_flip=True)

# prepare iterator
it = datagen.flow(samples, batch_size=batch_size)

# generate samples and plot
for i in range(20):

	# generate batch of images
	batch = it.next()

	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')

	# plot raw pixel data
	pyplot.imsave(f"train/person0/person0_{a}.jpg", image);a += 1


# --- Flip

# create image data augmentation generator
datagen = ImageDataGenerator(rotation_range=90)

# prepare iterator
it = datagen.flow(samples, batch_size=batch_size)

# generate samples and plot
for i in range(20):

	# generate batch of images
	batch = it.next()

	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')

	# plot raw pixel data
	pyplot.imsave(f"train/person0/person0_{a}.jpg", image);a += 1

# create image data augmentation generator
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# prepare iterator
it = datagen.flow(samples, batch_size=batch_size)

# generate samples and plot
for i in range(20):

	# generate batch of images
	batch = it.next()

	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')

	# plot raw pixel data
	pyplot.imsave(f"train/person0/person0_{a}.jpg", image);a += 1

# create image data augmentation generator
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
# prepare iterator
it = datagen.flow(samples, batch_size=batch_size)

# generate samples and plot
for i in range(20):

  # generate batch of images
  batch = it.next()

  # convert to unsigned integers for viewing
  image = batch[0].astype('uint8')

  a += 1

  # plot raw pixel data
  pyplot.imsave(f"train/person0/person0_{a}.jpg", image)

