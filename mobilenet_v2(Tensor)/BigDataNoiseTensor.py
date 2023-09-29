import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2 as cv

# libraries for more options:
# import math
# from skimage.transform import resize
# import skimage
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
l2 = ["b", "g", "r"]  # b: black, g: gray, r: red - loop path string
l3 = list(range(1, 11))  # 1-10 - loop path numbers
l4 = [round(float(i) * 0.1, 2) for i in range(0, 11)]  # standard deviation - loop path numbers 0.1-1.0
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:
                path = 'C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\data_all\\' + i + "_" + j + "_" + str(
                    k) + ".jpg"  # build the path to the image by lists
                image_ = cv.imread(path)  # read the image by path
                # Convert the image to float32
                image_ = np.float32(image_) / 255.0
                # Set the mean and standard deviation for Gaussian noise
                mean = 0
                stddev = m
                # Generate Gaussian noise
                noise = np.random.normal(mean, stddev, image_.shape).astype(np.float32)

                # Add noise to the image
                noisy_image = cv.add(image_, noise)

                # Clip the pixel values to the range [0, 1]
                noisy_image = np.clip(noisy_image, 0, 1)

                # Convert the noisy image back to uint8 format
                noisy_image = (noisy_image * 255).astype(np.uint8)

                # Save the noisy image
                save_path = 'noisy_image.jpg'
                cv.imwrite(save_path, noisy_image)

                predicted_label_, predicted_probability_ = predict_objects(save_path)

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
                                data_2_file = i + "_" + j + "_" + str(k) + " cat " + " STD: " + str(
                                    stddev) + " probability " + str(predicted_probability_) + "\n"
                                with open("data_noise_Tensor.txt", "a") as file2:
                                    file2.write(data_2_file)
                            if i == "d":
                                data_2_file = i + "_" + j + "_" + str(k) + " dog " + " STD: " + str(
                                    stddev) + " probability " + str(predicted_probability_) + "\n"
                                with open("data_noise_Tensor.txt", "a") as file2:
                                    file2.write(data_2_file)
