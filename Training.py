import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.compat.v1.disable_eager_execution()

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(
    128, 128, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='sigmoid'))

classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    '/Users/rabby/Desktop/Leaf-Disease/Dataset/train', target_size=(128, 128), batch_size=6, class_mode='categorical')
valid_set = test_datagen.flow_from_directory(
    '/Users/rabby/Desktop/Leaf-Disease/Dataset/val', target_size=(128, 128), batch_size=3, class_mode='categorical')

labels = (training_set.class_indices)
print(labels)


classifier.fit(training_set, steps_per_epoch=20,
               epochs=35, validation_data=valid_set)

classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)

    classifier.save_weights("my_model_weights.h5")
    classifier.save("model.h5")
    print("Saved model to disk")
