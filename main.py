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
classifier.add(Conv2D(8, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))
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
classifier.add(Dense(units = 224, activation = 'relu'))
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

training_set = train_datagen.flow_from_directory('G:/Pythn project/Image matching/Balnc/Code/TC2/Ashik/train/',
target_size = (224, 224),
batch_size = 5,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

validation_set = valid_datagen.flow_from_directory('G:/Pythn project/Image matching/Balnc/Code/TC2/Ashik/validation/',
target_size = (224, 224),
batch_size = 5,
shuffle = True,
color_mode="rgb",
class_mode = 'categorical')

# test_set = test_datagen.flow_from_directory('./validation_cut_pics',
# target_size = (224, 224),
# batch_size = 32,
# shuffle = True,
# class_mode = 'categorical')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


a = datetime.datetime.now().replace(microsecond=0)



history = classifier.fit_generator(training_set,
steps_per_epoch = 1000,
epochs = 2,
validation_data = validation_set,
validation_steps = 100)
classifier.save('./model_1000_32.h5')

b = datetime.datetime.now().replace(microsecond=0)
print(b-a)
plt.plot(history.history['acc'])

# Accuracy

plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

# Loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Part 3 - Making new predictions
# import numpy as np
# classifier = load_model('./models/second_try.h5')
# from keras.preprocessing import image
# test_image = image.load_img('./test images/testimage2.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# print(result)
# print(classes[np.argmax(result)])

# training_set.class_indices
# if result[0][0] == 1:
# 	prediction = 'person'
# 	print(prediction)
# else:
# 	prediction = 'non-person'
# 	print(prediction)


