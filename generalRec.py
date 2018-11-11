from keras.models import Sequential,Model
from keras.layers import Conv2D,Input
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Concatenate,Add
import matplotlib.pyplot as plt
import datetime

classes = ['DR','FT','LP']

# classifier1 = Sequential()
# input1 = Input(shape=(128, 128, 3))
# classifier1.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
# classifier1.add(MaxPooling2D(pool_size = (2, 2)))
# classifier1.add(Conv2D(16, (3, 3), activation = 'relu'))
# classifier1.add(MaxPooling2D(pool_size = (2, 2)))
# classifier1.add(Conv2D(64, (3, 3), activation = 'relu'))
# classifier1.add(MaxPooling2D(pool_size = (2, 2)))
# classifier1.add(Flatten())
# classifier1.add(Dense(units = 128, activation = 'relu'))
# classifier1.add(Dense(units = 3, activation = 'softmax'))
# classifier1.load_weights('./models/model_100_5_tc1.h5')

# classifier2 = Sequential()
# input2 = Input(shape=(128, 128, 3))
# classifier2.add(Conv2D(8, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
# classifier2.add(MaxPooling2D(pool_size = (2, 2)))
# classifier2.add(Conv2D(16, (3, 3), activation = 'relu'))
# classifier2.add(MaxPooling2D(pool_size = (2, 2)))
# classifier2.add(Conv2D(64, (3, 3), activation = 'relu'))
# classifier2.add(MaxPooling2D(pool_size = (2, 2)))
# classifier2.add(Flatten())
# classifier2.add(Dense(units = 128, activation = 'relu'))
# classifier2.add(Dense(units = 3, activation = 'softmax'))
# classifier2.load_weights('./models/model_100_5_tc2.h5')


model1 = Sequential()
model1=load_model('./models/rec1/rec1_model.h5')
model2= Sequential()
model2 = load_model('./models/rec2/rec2_model.h5')
# Consolidated model
consolidated_model = Model(
    inputs=[model1.input,model2.input],
    outputs=[model2.output]
)
consolidated_model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
consolidated_model.save('./models/general_model.h5')