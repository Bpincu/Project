import Model

import os
import keras
from keras_preprocessing.image import ImageDataGenerator
import PIL
import numpy as np 
from PIL import Image
import sys

def folderToTest(path): #Will make a full folder into a 20% one
    l = os.listdir(path)
    counter = 0
    for img in l:
        if(counter == 4):
            counter = 0
        else:
            counter += 1
            os.remove(os.path.join(path, img))

def predictImg(model,img): #Will predict the image with the model
    predictions = model.predict(np.array([np.array(img)]))
    return predictions[0]

def showMenu(): #Shows the menu and gets input
    print("Welcome To Bar's Project, Choose What You Desire")
    print("1. Load The Model")
    print("2. Test An Image")
    print("3. Train The Model")
    print("4. Exit")
    x = input()
    while not (x.isdigit() and (int(x) > 0) and (int(x) < 5)):
        print("This is not a valid input")
        x = input()
    return int(x)

def train_model(): #Will train the model
    datagen = ImageDataGenerator(rescale = 1/255.0, horizontal_flip= True)
    trainGenerator = datagen.flow_from_directory('./Train', target_size=(160,120), color_mode='rgb',batch_size = 1)
    testGenerator = datagen.flow_from_directory('./Test', target_size=(160,120), color_mode='rgb', batch_size = 1)
    model = Model.define_model()
    model.fit_generator(trainGenerator,steps_per_epoch=len(trainGenerator),validation_data=testGenerator, epochs=3, verbose = 1)
    test_loss, test_acc = model.evaluate(trainGenerator, verbose=1)
    model.save("ClothModel.h5")
    print('\nTest accuracy:', test_acc)
    return model

def load_model(): #Loading the saved model
    model = Model.define_model()
    model.load_weights("ClothModel.h5")
    return model

def load_photo(img_path, img_width, img_height): #Will load photo from path, minimize it and convert it to rgb
    image = Image.open(img_path)
    image = image.resize((img_width, img_height))
    image = image.convert("RGB")
    return image

def predict(model): #Will predict the image with the model
    classes = ["Backpack", "Bag", "Flipflop", "Glasses", "Shoe", "T-Shirt", "Watch"]
    datagen = ImageDataGenerator(rescale = 1/255.0)
    s = input()
    while not os.path.isfile(s):
        print("Directory not exists.")
        s = input()
    img = load_photo(s,120,160)
    predictions_single = predictImg(model,img)
    print(predictions_single)
    testGenerator = datagen.flow_from_directory('./Test', target_size=(160,120), color_mode='rgb', batch_size = 1)
    print(testGenerator.class_indices)
    print(classes[np.argmax(predictions_single)])


def main():
    os.environ['PATH'] += ';' + "E:\\Old\\cudnn-10.2-windows10-x64-v7.6.5.32\\cuda\\bin"
    print("E:\\Old\\cudnn-10.2-windows10-x64-v7.6.5.32\\cuda\\bin")
    choise = showMenu()
    model_loaded = False
    while not choise == 4:
        if choise == 1:
            model = load_model()
            model_loaded = True
        if choise == 2:
            if not model_loaded:
                print("\nPlease Load The Model First")
            else:
                predict(model)            
        if choise == 3:
            model = train_model()
            model_loaded = True
        choise = showMenu()
    sys.exit()

if __name__ == "__main__":
    main()