from keras.models import Sequential,Model
from keras.layers import Conv2D,Input
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Concatenate,Add
import matplotlib.pyplot as plt
import datetime
# Initialising the CNN
classes = ['DR','FT','LP']

classifier1 = Sequential()
input1 = Input(shape=(128, 128, 3))
# Step 1 - Convolution
classifier1.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
# Step 2 - Pooling
classifier1.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier1.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2, 2)))

classifier1.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier1.add(Flatten())
# Step 4 - Full connection
classifier1.add(Dense(units = 128, activation = 'relu'))
classifier1.add(Dense(units = 3, activation = 'softmax'))

classifier1.load_weights('./models/model_100_5_tc1.h5')


classifier2 = Sequential()
input2 = Input(shape=(128, 128, 3))
# Step 1 - Convolution
classifier2.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
# Step 2 - Pooling
classifier2.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier2.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))

classifier2.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier2.add(Flatten())
# Step 4 - Full connection
classifier2.add(Dense(units = 128, activation = 'relu'))
classifier2.add(Dense(units = 3, activation = 'softmax'))

classifier2.load_weights('./models/model_100_5_tc2.h5')


consolidated_model = Model(
    inputs=[classifier1.input,classifier2.input],
    outputs=[classifier2.output]
)
consolidated_model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


consolidated_model.save('./models/rec1_model.h5')


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


