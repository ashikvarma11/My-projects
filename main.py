from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import datetime
# Initialising the CNN
classes = ['DR','FT','LP']

classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

# optimiz = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)

valid_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./TC1/ABCD/train/',
target_size = (128, 128),
batch_size = 5,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

validation_set = valid_datagen.flow_from_directory('./TC1/ABCD/validation/',
target_size = (128, 128),
batch_size = 5,
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
classifier.save_weights('./models/model_100_5_tc1.h5')

