
import keras
import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import os
import numpy as np
import parser

import numpy as np


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

classes = ['DR','FT','LP']
classifier = load_model('./models/rec1_model.h5')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.preprocessing import image
IMAGE_FOLDER = './TC1/ABCD/validation/LP/'
images = os.listdir(IMAGE_FOLDER)
count=0


# test_image = image.load_img(IMAGE_FOLDER+ images[0], target_size = (128, 128))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# print(images[0] + " -> " + classes[np.argmax(result)])
# print(result)




for filename in images:

	test_image = image.load_img(IMAGE_FOLDER+ filename, target_size = (128, 128))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)
	print(filename + " -> " + classes[np.argmax(result)])
	if classes[np.argmax(result)] == 'DR':
		count+=1
print("\n correct count = "+ str(count) +"\n total count = "+str(len(images))+"\n")
print("Percentage:" + str((count)/len(images)*100)+"%")
print("\n")
