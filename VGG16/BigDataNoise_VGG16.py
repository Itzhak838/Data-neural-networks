"""
the following code is used to detect a big data of images
its detect the images and save the data to a file by name, confidence and noise level
in this case, the images are in format: c/d_b/g/r_0.1-1.jpg
the images are in the folder "inference/images/data_all"
the output is in the file "Data_Noise_VGG16.txt"
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
l4 = [round(float(i)*0.1, 2) for i in range(0, 11)]  # standard deviation - loop path numbers 0.1-1.0
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:

                path = "C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\data_all\\" + i + "_" + j + "_" + str(k) + ".jpg"
                image_start = cv.imread(path)  # read the image by path
                img_path = "noisy_image.jpg"
                stddev = m
                if m != 0:
                    image_start = np.float32(image_start) / 255.0
                    # Set the mean and standard deviation for Gaussian noise
                    mean = 0
                    # Generate Gaussian noise
                    noise = np.random.normal(mean, stddev, image_start.shape).astype(np.float32)

                    # Add noise to the image
                    noisy_image = cv.add(image_start, noise)

                    # Clip the pixel values to the range [0, 1]
                    noisy_image = np.clip(noisy_image, 0, 1)

                    # Convert the noisy image back to uint8 format
                    noisy_image = (noisy_image * 255).astype(np.uint8)

                    # rescaled_image = skimage.img_as_ubyte(rescaled_image)  # convert the image to uint8
                    cv.imwrite(img_path, noisy_image)  # save the image to print it later
                else:
                    cv.imwrite(img_path, image_start)

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
                    data_2_file = i + "_" + j + "_" + str(k) + " real: cat, predicted: " + label[1] + " " + str(label[2]) + " STD: " + str(stddev) + "\n"
                elif i == "d":
                    data_2_file = i + "_" + j + "_" + str(k) + " real: dog, predicted: " + label[1] + " " + str(label[2]) + " STD: " + str(stddev) + "\n"
                else:
                    data_2_file = "Error: wrong string"

                print(data_2_file)
                with open("Data_Noise_VGG16.txt", "a") as file:
                    file.write(data_2_file)
















