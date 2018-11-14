import keras
from keras.models import load_model
import os
import numpy as np
import parser
import datetime

classes = ['DR','FT','LP']
postures = ['back','front','left','right']
classifier = load_model('../models/rec1/rec1_model.h5')
classifier2 = load_model('../models/rec1/rec1_modelMatch_FT.h5')
classifier3 = load_model('../models/rec1/rec1_modelMatch_DR.h5')
classifier4 = load_model('../models/rec1/rec1_modelMatch_LP.h5')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing import image
IMAGE_FOLDER = '../TC2/Ashik/train/FT/'
# images = os.listdir(IMAGE_FOLDER)
# count=0
# a = datetime.datetime.now().replace(microsecond=0)
# for filename in images:
# 	test_image = image.load_img(IMAGE_FOLDER+ filename, target_size = (128, 128))
# 	test_image = image.img_to_array(test_image)
# 	test_image = np.expand_dims(test_image, axis = 0)
# 	result = classifier.predict(test_image)
# 	print(filename + " -> " + classes[np.argmax(result)])
# 	if classes[np.argmax(result)] == 'FT':
# 		count+=1
# 		result2 = classifier2.predict(test_image)
# 		print(" Image model FT -> " + postures[np.argmax(result2)])
# 	elif classes[np.argmax(result)] == 'DR':
# 		result2 = classifier3.predict(test_image)
# 		print(" Image model DR-> " + postures[np.argmax(result2)])
# 	else:
# 		result2 = classifier4.predict(test_image)
# 		print(" Image model LP-> " + postures[np.argmax(result2)])
# print("\n correct count = "+ str(count) +"\n total count = "+str(len(images))+"\n")
# print("Percentage:" + str((count)/len(images)*100)+"%")
# print("\n")
# b = datetime.datetime.now().replace(microsecond=0)
# print("time for testing: " + str(b-a) )

filename = 'VID_20181109_162242 262.jpg'


test_image = image.load_img(IMAGE_FOLDER+filename, target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(filename + " -> " + classes[np.argmax(result)])
if classes[np.argmax(result)] == 'FT':
	result2 = classifier2.predict(test_image)
	print(" Image model FT -> " + postures[np.argmax(result2)])
elif classes[np.argmax(result)] == 'DR':
	result2 = classifier3.predict(test_image)
	print(" Image model DR-> " + postures[np.argmax(result2)])
else:
	result2 = classifier4.predict(test_image)
	print(" Image model LP-> " + postures[np.argmax(result2)])


