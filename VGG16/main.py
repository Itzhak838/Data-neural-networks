"""
The code designed to predict an image objects in the VGG16 model
"""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
# import os
# from os import listdir
# from PIL import Image as PImage

img_width, img_height = 224, 224

model_pretrained = VGG16(weights='imagenet',
                         include_top=True,
                         input_shape=(img_height, img_width, 3))

# Insert the correct path of your image below
img_path = "C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\data_all\\c_r_2.jpg"
img = image.load_img(img_path, target_size=(img_width, img_height))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

#predict the result
cnn_feature = model_pretrained.predict(img_data,verbose=0)
# decode the results into a list of tuples (class, description, probability)
label = decode_predictions(cnn_feature)
label = label[0][0]

# plt.imshow(img)

stringprint = "%.1f" % round(label[2] * 100, 1)
# plt.title(label[1] + " " + str(stringprint) + "%", fontsize=20)
# print(float(stringprint)/100)
# plt.axis('off')
# plt.show()
