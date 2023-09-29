"""
This script design to adding noise to all channels together (white and black noise)
 and then predicts on it using a pretrained YOLOv8n model.
 The target is to check the resolution limit of the data.
"""
import cv2 as cv
from ultralytics import YOLO
import numpy as np
l1 = ["d"]  # d: dog - loop path string -  dog
l2 = ["b", "g", "r"]  # b: black, g: green, r: red - loop path string
l3 = list(range(1, 11))  # 1-10 - loop path numbers
l4 = [round(float(i)*0.1, 2) for i in range(0, 11)]  # standard deviation - loop path numbers 0.1-1.0
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:
                write_data = False  # check if the specific data was written to the file
                path = "inference/images/data_all/" + i + "_" + j + "_" + str(k) + ".jpg"  # build the path to the image by lists
                image = cv.imread(path)  # read the image by path
                # Convert the image to float32
                image = np.float32(image) / 255.0
                # Set the mean and standard deviation for Gaussian noise
                mean = 0
                stddev = m
                # Generate Gaussian noise
                noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)

                # Add noise to the image
                noisy_image = cv.add(image, noise)

                # Clip the pixel values to the range [0, 1]
                noisy_image = np.clip(noisy_image, 0, 1)

                # Convert the noisy image back to uint8 format
                noisy_image = (noisy_image * 255).astype(np.uint8)

                # Save the noisy image
                save_path = 'noisy_image.jpg'
                cv.imwrite(save_path, noisy_image)

                # load a pretrained YOLOv8n model
                model = YOLO("yolov8n.pt", "v8")

                # predict the compressed image
                detection_output = model.predict(save_path, conf=0.05, save=False)

                # Loop through the detection output to save the data to a file by name and confidence
                # the data is saved in the file "p_data.txt"
                for r in detection_output:
                    boxes = r.boxes
                    for box in boxes:
                        conf = box.conf.numpy()[0]  # get the probability of the bounding box
                        name = int(box.cls.numpy()[0])  # get the class number of the bounding box
                        if (name == 16) and (write_data is False):  # check if the class is a dog
                            data_2_file = i + "_" + j + "_" + str(k) + " STD: " + str(stddev) + " probability: " + str(conf) + "\n"
                            print(data_2_file)
                            with open("noise_data.txt", "a") as file:
                                file.write(data_2_file)
                                write_data = True
