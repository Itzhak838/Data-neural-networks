import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
# libraries for more options:
# import math
# from skimage.transform import resize
# import skimage
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


l1 = ["d"]  # c: cat, d: dog - loop path string
l2 = ["b", "g", "r"]  # b: black, g: green, r: red - loop path string
l3 = [1, 2]  # 1-10 - loop path numbers
l4 = [0, 2, 4, 6, 8, 10]  # compression ratio loop
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:
                image_b = "C:/Itzhak/BSc/4yr/Final_Project/Neural Network results/Blur/" + i + "_" + j + "_" + str(k) + "_" + str(m) + ".jpg"  # build the path to the image by lists

                image_b_o = cv.imread(image_b)  # read the image by path
                predicted_label_, predicted_probability_ = predict_objects(image_b)


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
                            if i == "d":
                                data_2_file = i + "_" + j + "_" + str(k) + "_" + str(m) + " " + str(predicted_probability_) + "\n"
                                with open("Blur_data.txt", "a") as file2:
                                    file2.write(data_2_file)

# Display the image
# img = Image.open(image_path)
# plt.imshow(img)
# plt.axis('off')
# plt.show()
