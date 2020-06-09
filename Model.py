
import keras
from keras import Sequential
from keras.layers.core import Dropout
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization

def define_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(160,120,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return model
