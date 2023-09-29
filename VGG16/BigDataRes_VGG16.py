"""
the following code is used to detect a big data of images
its detect the images and save the data to a file by name, confidence and compression ratio
in this case, the images are in format: c/d_b/g/r_1-10.jpg
the images are in the folder "inference/images/data_all"
the output is in the file "Data_VGG16.txt"
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
l1 = ["c", "d"]  # c: cat, d: dog - loop path string
l2 = ["b", "g", "r"]  # b: black, g: gray, r: red - loop path string
l3 = list(range(1, 11))  # 1-10 - loop path numbers
l4 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # compression ratio loop
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:

                path = "C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\data_all\\" + i + "_" + j + "_" + str(k) + ".jpg"
                image_input = cv.imread(path)  # read the image by path
                pixels_2_one_pixel = m  # compression ratio
                scale = 1/math.sqrt(pixels_2_one_pixel)  # calculate the scale factor to resize the image by the compression ratio
                resized_image = resize(image_input, (image_input.shape[0] * scale, image_input.shape[1] * scale), mode='constant', anti_aliasing=False)  # resize the image by the scale factor
                rescaled_image = resize(resized_image, (image_input.shape[0], image_input.shape[1]), anti_aliasing=False)  # rescale the image to the original size
                rescaled_image = skimage.img_as_ubyte(rescaled_image)  # convert the image to uint8
                img_path = "C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\resized.jpg"
                cv.imwrite(img_path, rescaled_image)  # save the image to print it later
                img = image.load_img(img_path, target_size=(img_width, img_height))
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
                with open("Data_VGG16.txt", "a") as file:
                    file.write(data_2_file)
















