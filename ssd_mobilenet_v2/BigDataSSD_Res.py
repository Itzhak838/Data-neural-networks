"""
the following code and the other scripts is used to detect a big data of images.
its detect the images and save the data to a file by name, confidence and compression ratio
in this case, the images are in format: c/d_b/g/r_1-10.jpg
the images are in the folder "inference/images/data_all"
the output is in the file "p_data_SSD.txt"
"""
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import math
from skimage.transform import resize
import skimage
# import numpy as np


def load_and_preprocess_image(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(320, 320))
    image = tf.keras.utils.img_to_array(image)
    image = tf.image.convert_image_dtype(image, tf.uint8)  # Convert to uint8
    return image


# Function to perform object detection
def detect_objects(image_path):
    image = load_and_preprocess_image(image_path)
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    detections = model(image)
    # Extract the detected boxes, classes, and scores
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()
    return boxes, classes, scores


model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
model = hub.load(model_url)
l1 = ["c", "d"]  # c: cat, d: dog - loop path string
l2 = ["b", "g", "r"]  # b: black, g: gray, r: red - loop path string
l3 = list(range(1, 11))  # 1-10 - loop path numbers
l4 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # compression ratio loop
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:
                image_path_ = "C:/Itzhak/BSc/4yr/Final_Project/pythonProject/yolov8-silva/inference/images/data_all/" + i + "_" + j + "_" + str(
                    k) + ".jpg"  # build the path to the image by lists
                image_input = cv.imread(image_path_)  # read the image by path
                pixels_2_one_pixel = m  # compression ratio
                scale = 1 / math.sqrt(pixels_2_one_pixel)  # calculate the scale factor to resize the image by the compression ratio
                resized_image = resize(image_input, (image_input.shape[0] * scale, image_input.shape[1] * scale),
                                       mode='constant', anti_aliasing=False)  # resize the image by the scale factor
                rescaled_image = resize(resized_image, (image_input.shape[0], image_input.shape[1]),
                                        anti_aliasing=False)  # rescale the image to the original size
                rescaled_image = skimage.img_as_ubyte(rescaled_image)  # convert the image to uint8
                comp_path = "C:/Itzhak/BSc/4yr/Final_Project/pythonProject/yolov8-silva/inference/images/resized.jpg"
                cv.imwrite(comp_path, rescaled_image)  # save the image to print it later
                boxes_, classes_, scores_ = detect_objects(comp_path)
                # Display the image with bounding boxes
                img = Image.open(comp_path)
                plt.imshow(img)
                ax = plt.gca()
                # Threshold for displaying detected objects
                confidence_threshold = 0.05
                num_of_loop = 0
                for box, cls, score in zip(boxes_, classes_, scores_):
                    num_of_loop += 1
                    if score > confidence_threshold:
                        y_min, x_min, y_max, x_max = box
                        width, height = img.size  # Get width and height using .size
                        y_min, x_min, y_max, x_max = (
                            int(y_min * height),
                            int(x_min * width),
                            int(y_max * height),
                            int(x_max * width),
                        )
                        print("THIS CLS IS: ", cls, "THIS SCORE IS: ", score, " num of loop: ", num_of_loop)

                        if num_of_loop == 1:
                            if (cls == 18) and (i == "d"):
                                data_2_file = i + "_" + j + "_" + str(k) + "_" + str(m) + " dog " + str(score) + "\n"
                                with open("p_data_SSD.txt", "a") as file:
                                    file.write(data_2_file)

                            if (cls == 17) and (i == "c"):
                                data_2_file = i + "_" + j + "_" + str(k) + "_" + str(m) + " cat " + str(score) + "\n"
                                with open("p_data_SSD.txt", "a") as file:
                                    file.write(data_2_file)
