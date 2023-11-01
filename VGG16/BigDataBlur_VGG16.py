"""
the following code is used to detect a big data of images
its detect the images and save the data to a file by name, confidence and blur level
in this case, the images are in format: c/d_b/g/r_0-10.jpg
the output is in the file "Blur_VGG16.txt"
"""
import cv2 as cv
import math
from skimage.transform import resize
import skimage
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

img_width, img_height = 224, 224

model_pretrained = VGG16(weights='imagenet',
                         include_top=True,
                         input_shape=(img_height, img_width, 3))
l1 = ["d"]  # c: cat, d: dog - loop path string
l2 = ["b", "g", "r"]  # b: black, g: green, r: red - loop path string
l3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 1-10 - loop path numbers
l4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # standard deviation - blur level
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:

                path = "C:/Itzhak/BSc/4yr/Final_Project/pythonProject/yolov8-silva/inference/images/data_all/" + i + "_" + j + "_" + str(
                    k) + ".jpg"  # build the path to the image by lists
                image_input = cv.imread(path)  # read the image by path
                if m == 0:  # if the blur level is 0, predict on the original image
                    img = image.load_img(path, target_size=(img_width, img_height))
                    img_data = image.img_to_array(img)
                    img_data = np.expand_dims(img_data, axis=0)
                    img_data = preprocess_input(img_data)
                    # predict the result
                    cnn_feature = model_pretrained.predict(img_data, verbose=0)
                    # decode the results into a list of tuples (class, description, probability)
                    label = decode_predictions(cnn_feature)
                    label = label[0][0]
                else:  # if the blur level is not 0, predict on the blured image
                    sigma = m  # standard deviation
                    blurred = cv.GaussianBlur(image_input, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    wr_path = 'blurred.jpg'
                    blurred_image = cv.imwrite(wr_path, blurred)
                    img = image.load_img(wr_path, target_size=(img_width, img_height))
                    img_data = image.img_to_array(img)
                    img_data = np.expand_dims(img_data, axis=0)
                    img_data = preprocess_input(img_data)
                    # predict the result
                    cnn_feature = model_pretrained.predict(img_data, verbose=0)
                    # decode the results into a list of tuples (class, description, probability)
                    label = decode_predictions(cnn_feature)
                    label = label[0][0]
                # stringprint = "%.1f" % round(label[2] * 100, 1)
                # print(label[1] + " ", float(stringprint) / 100, label[2])
                if i == "c":
                    data_2_file = i + "_" + j + "_" + str(k) + "_" + str(m) + " real: cat, predicted: " + label[1] + " " + str(label[2]) + "\n"
                elif i == "d":
                    data_2_file = i + "_" + j + "_" + str(k) + "_" + str(m) + " real: dog, predicted: " + label[1] + " " + str(label[2]) + "\n"

                print(data_2_file)
                with open("Blur_VGG16.txt", "a") as file:
                    file.write(data_2_file)
















