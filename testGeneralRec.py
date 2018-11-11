import keras
from keras.models import load_model
import os
import numpy as np
import parser

classes = ['DR','FT','LP']
classifier = load_model('./models/general_model.h5')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing import image
IMAGE_FOLDER = './TC1/ABCD/validation/LP/'
images = os.listdir(IMAGE_FOLDER)
count=0
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
