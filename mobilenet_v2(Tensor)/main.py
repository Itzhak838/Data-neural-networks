"""
the following code and the other scripts is used to detect a big data of images.
its detect the images and save the data to a file by name, confidence and compression ratio
in this case, the images are in format: c/d_b/g/r_1-10.jpg
the images are in the folder "inference/images/data_all"
the output is in the file "p_data.txt"
"""
import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import math
from skimage.transform import resize
import skimage
# libraries for more options:
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# Load a pre-trained MobileNetV2 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)


# Load and preprocess an image
def load_and_preprocess_image(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image


# Function to predict objects in an image
def predict_objects(image_path):
    image = load_and_preprocess_image(image_path)
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model(image)

    # Load the ImageNet class labels
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    with open(labels_path) as f:
        labels = f.read().splitlines()

    # Get the top predicted class index and its corresponding probability
    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_probability = tf.nn.softmax(predictions)[0][predicted_class_index].numpy()
    predicted_label = labels[predicted_class_index]

    return predicted_label, predicted_probability


l1 = ["c", "d"]  # c: cat, d: dog - loop path string
l2 = ["b", "g", "r"]  # b: black, g: gray, r: red - loop path string
l3 = list(range(1, 11))  # 1-10 - loop path numbers
l4 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # compression ratio loop
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:
                img_path = 'C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\data_all\\' + i + "_" + j + "_" + str(k) + ".jpg"  # build the path to the image by lists
                image_input = cv.imread(img_path)  # read the image by path
                pixels_2_one_pixel = m  # compression ratio
                scale = 1 / math.sqrt(
                    pixels_2_one_pixel)  # calculate the scale factor to resize the image by the compression ratio
                resized_image = resize(image_input, (image_input.shape[0] * scale, image_input.shape[1] * scale),
                                       mode='constant', anti_aliasing=False)  # resize the image by the scale factor
                rescaled_image = resize(resized_image, (image_input.shape[0], image_input.shape[1]),
                                        anti_aliasing=False)  # rescale the image to the original size
                rescaled_image = skimage.img_as_ubyte(rescaled_image)  # convert the image to uint8
                comp_img_path = "C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\resized.jpg"
                cv.imwrite(comp_img_path, rescaled_image)
                predicted_label_, predicted_probability_ = predict_objects(comp_img_path)

                # Display the top predicted object and its probability
                print(i + "_" + j + "_" + str(k) + "_" + str(m), predicted_label_, predicted_probability_)

                if i == "c":
                    file_load_path = 'cats.txt'
                elif i == "d":
                    file_load_path = 'dogs.txt'
                else:
                    raise Exception("Error: wrong path or image file name")

                with open(file_load_path, 'r') as file1:
                    for line in file1:
                        line = line.strip()
                        if line == predicted_label_:
                            if i == "c":
                                data_2_file = i + "_" + j + "_" + str(k) + "_" + str(m) + " cat " + str(predicted_probability_) + "\n"
                                with open("big_data.txt", "a") as file2:
                                    file2.write(data_2_file)
                            if i == "d":
                                data_2_file = i + "_" + j + "_" + str(k) + "_" + str(m) + " dog " + str(predicted_probability_) + "\n"
                                with open("big_data.txt", "a") as file2:
                                    file2.write(data_2_file)
