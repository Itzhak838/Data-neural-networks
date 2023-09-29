"""
the following code is used to detect a big data of images
its detect the images and save the data to a file by name, confidence and square dimension
in this case, the images are in format: c/d_b/g/r_1-10.jpg
the images are in the folder "inference/images/data_all"
the output is in the file "Data_Square_VGG16.txt"
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
l2 = ["b", "g", "r"]  # b: black, g: gray, r: red - loop path string
l3 = list(range(1, 11))  # 1-10 - loop path numbers
l4 = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]  # n*n black square
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:
                path = "C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\data_all\\" + i + "_" + j + "_" + str(k) + ".jpg"
                img_path = "noisy_image.jpg"
                img = cv.imread(path)  # read the image by path
                x_center = img.shape[0] // 2
                y_center = img.shape[1] // 2
                # print("x_center: ", x_center, "y_center: ", y_center)
                # adding a stop condition when the square is out of the image
                if (x_center - m >= 0) and (y_center - m >= 0) and (x_center + m // 2 <= img.shape[0]) and (
                        y_center + m // 2 <= img.shape[1]):
                    img[x_center - m // 2:x_center + m // 2, y_center - m // 2:y_center + m // 2] = 0, 0, 0
                    cv.imwrite(img_path, img)  # save the image to print it later
                    img_load = image.load_img(img_path, target_size=(img_width, img_height))
                    img_data = image.img_to_array(img_load)
                    img_data = np.expand_dims(img_data, axis=0)
                    img_data = preprocess_input(img_data)
                    # predict the result
                    cnn_feature = model_pretrained.predict(img_data, verbose=0)
                    # decode the results into a list of tuples (class, description, probability)
                    label = decode_predictions(cnn_feature)
                    label = label[0][0]
                    if i == "c":
                        data_2_file = i + "_" + j + "_" + str(k) + " real: cat, predicted: " + label[1] + " " + str(label[2]) + " Dimension of noise: " + str(m) + '*' + str(m) + "\n"
                    elif i == "d":
                        data_2_file = i + "_" + j + "_" + str(k) + " real: dog, predicted: " + label[1] + " " + str(label[2]) + " Dimension of noise: " + str(m) + '*' + str(m) + "\n"
                    else:
                        data_2_file = "Error: wrong string"

                    print(data_2_file)
                    with open("Data_Square_VGG16.txt", "a") as file:
                        file.write(data_2_file)
















