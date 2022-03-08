from keras.applications import vgg16
from tensorflow.keras import layers, models

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model

from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

def add_layer(bottom_model, num_classes):
  top_model = bottom_model.output
  top_model = GlobalAveragePooling2D()(top_model)
  top_model = Dense(1024, activation="relu")(top_model)
  top_model = Dense(512, activation="relu")(top_model)
  top_model = Dense(num_classes, activation="softmax")(top_model)
  return top_model


if __name__ == "__main__":
  img_rows = 150; img_cols = 150

  model = vgg16.VGG16(weights = 'imagenet',
                      include_top = False,
                      input_shape = (img_rows, img_cols, 3))

  for layer in model.layers:
    layer.trainable = False

  # What exactly is this ?  
  num_classes = 2

  FC_Head = add_layer(model, num_classes) 

  model = Model(inputs = model.input, outputs=FC_Head)

  model.summary()

  train_data_dir = "./train/"
  validation_data_dir = "./test/"

  train_datagen = ImageDataGenerator(
                  rotation_range=45,
                  width_shift_range=0.3,
                  height_shift_range=0.3,
                  horizontal_flip=True,
                  fill_mode="nearest")

  validation_datagen = ImageDataGenerator()

  batch_size = 1

  train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

  validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
  
  checkpoint = ModelCheckpoint("face_detector.h15",
                              monitor="val_loss",
                              mode="min",
                              save_best_only=True,
                              verbose=1)

  earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            restore_best_weights=True)

  callbacks = [earlystop, checkpoint]

  model.compile(loss="categorical_crossentropy",
                optimizer="Adam",
                metrics=["accuracy"])

  nb_train_samples = 100

  epochs = 1
  batch_size = 1
  
  history = model.fit(
    train_generator,
    epochs = epochs,
    callbacks=callbacks,
    validation_data=validation_generator) 
