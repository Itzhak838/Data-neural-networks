"""
the following code and the other scripts is used to detect a big data of images.
its detect the images and save the data to a file by name, confidence and blur level.
in this case, the images are in format: c/d_b/g/r_1-10.jpg
the output is in the file "Blur_data_SSD.txt"
"""
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv


# libraries for more options:
# import math
# from skimage.transform import resize
# import skimage
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
l1 = ["d"]  # d: dog - loop path string
l2 = ["b", "g", "r"]  # b: black, g: green, r: red - loop path string
l3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 1-10 - loop path numbers
l4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # standard deviation - blur level
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:
                path = "C:/Itzhak/BSc/4yr/Final_Project/pythonProject/yolov8-silva/inference/images/data_all/" + i + "_" + j + "_" + str(k) + ".jpg"  # build the path to the image by lists
                image_input = cv.imread(path)  # read the image by path
                if m == 0:  # if the blur level is 0, predict on the original image
                    boxes_, classes_, scores_ = detect_objects(path)
                else:  # if the blur level is not 0, predict on the blured image
                    sigma = m  # standard deviation
                    blurred = cv.GaussianBlur(image_input, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    wr_path = 'blurred.jpg'
                    blurred_image = cv.imwrite(wr_path, blurred)
                    boxes_, classes_, scores_ = detect_objects(wr_path)
                # Display the image with bounding boxes
                img = Image.open(path)
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
                        # adding the list object detected as a dog - dog is the biggest probability
                        if num_of_loop == 1:
                            if (cls == 18) and (i == "d"):
                                data_2_file = i + "_" + j + "_" + str(k) + "_" + str(m) + " " + str(score) + "\n"
                                with open("Blur_data_SSD.txt", "a") as file:
                                    file.write(data_2_file)
