from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import datetime

classes = ['back','front','left','right']

#classifier for DR
classifier = Sequential()
classifier.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'softmax'))
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)
valid_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../2dmod/DR',
target_size = (128, 128),
batch_size = 2,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

validation_set = valid_datagen.flow_from_directory('../2dmod/DR',
target_size = (128, 128),
batch_size = 2,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
a = datetime.datetime.now().replace(microsecond=0)

history = classifier.fit_generator(training_set,
steps_per_epoch = 100,
epochs = 2,
validation_data = validation_set,
validation_steps = 100)
classifier.save('../models/rec1/rec1_modelMatch_DR.h5')
b = datetime.datetime.now().replace(microsecond=0)
print("time for model image matching: " + str(b-a) )

#classifier for LP
classifier2 = Sequential()
classifier2.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))
classifier2.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))
classifier2.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))
classifier2.add(Flatten())
classifier2.add(Dense(units = 128, activation = 'relu'))
classifier2.add(Dense(units = 4, activation = 'softmax'))
classifier2.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)
valid_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../2dmod/LP',
target_size = (128, 128),
batch_size = 2,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

validation_set = valid_datagen.flow_from_directory('../2dmod/LP',
target_size = (128, 128),
batch_size = 2,
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
classifier2.save('../models/rec1/rec1_modelMatch_LP.h5')
b = datetime.datetime.now().replace(microsecond=0)
print("time for model image matching: " + str(b-a) )

#classifer for FT
classifier3 = Sequential()
classifier3.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier3.add(MaxPooling2D(pool_size = (2, 2)))
classifier3.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier3.add(MaxPooling2D(pool_size = (2, 2)))
classifier3.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier3.add(MaxPooling2D(pool_size = (2, 2)))
classifier3.add(Flatten())
classifier3.add(Dense(units = 128, activation = 'relu'))
classifier3.add(Dense(units = 4, activation = 'softmax'))
classifier3.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)
valid_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../2dmod/FT',
target_size = (128, 128),
batch_size = 2,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

validation_set = valid_datagen.flow_from_directory('../2dmod/FT',
target_size = (128, 128),
batch_size = 2,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
a = datetime.datetime.now().replace(microsecond=0)

history = classifier3.fit_generator(training_set,
steps_per_epoch = 100,
epochs = 2,
validation_data = validation_set,
validation_steps = 100)
classifier3.save('../models/rec1/rec1_modelMatch_FT.h5')
b = datetime.datetime.now().replace(microsecond=0)
print("time for model image matching: " + str(b-a) )