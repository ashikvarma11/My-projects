from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import datetime

classes = ['DR','FT','LP']

classifier1 = Sequential()
classifier1.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2, 2)))
classifier1.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2, 2)))
classifier1.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2, 2)))
classifier1.add(Flatten())
classifier1.add(Dense(units = 128, activation = 'relu'))
classifier1.add(Dense(units = 3, activation = 'softmax'))
classifier1.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)

valid_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../TC2/Arun/train/',
target_size = (128, 128),
batch_size = 5,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

validation_set = valid_datagen.flow_from_directory('../TC2/Arun/validation/',
target_size = (128, 128),
batch_size = 5,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
a = datetime.datetime.now().replace(microsecond=0)

history = classifier1.fit_generator(training_set,
steps_per_epoch = 100,
epochs = 2,
validation_data = validation_set,
validation_steps = 100)
classifier1.save_weights('../models/rec3/rec3_arun.h5')

classifier2 = Sequential()
classifier2.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))
classifier2.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))
classifier2.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))
classifier2.add(Flatten())
classifier2.add(Dense(units = 128, activation = 'relu'))
classifier2.add(Dense(units = 3, activation = 'softmax'))
classifier2.load_weights('../models/rec3/rec3_arun.h5')
classifier2.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)

valid_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../TC2/Ashik/train/',
target_size = (128, 128),
batch_size = 5,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

validation_set = valid_datagen.flow_from_directory('../TC2/Ashik/validation/',
target_size = (128, 128),
batch_size = 5,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

a = datetime.datetime.now().replace(microsecond=0)
history = classifier2.fit_generator(training_set,
steps_per_epoch = 100,
epochs = 2,
validation_data = validation_set,
validation_steps = 100)
classifier2.save_weights('../models/rec3/rec3_tc2.h5')