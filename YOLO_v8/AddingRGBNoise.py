"""
This script design to add a colord gaussian noise to an image in one channel
 and then predicts on it using a pretrained YOLOv8n model.
 The target is to check the resolution limit of the data.
"""
from ultralytics import YOLO
import numpy as np
import cv2 as cv
model = YOLO("yolov8n.pt", "v8")  # load a pretrained YOLOv8n model
path = "C:/Itzhak/BSc/4yr/Final_Project/pythonProject/yolov8-silva/inference/images/data_all/d_g_3.jpg"
image_input = cv.imread(path, cv.IMREAD_UNCHANGED)  # read the image as is
for i in range(0, 3):  # add noise to RGB channels 0-2
    for j in [range(0, 11)]:  # add noise with different Standard deviation 0-10
        write_data = False  # check if the specific data was written to the file
        STD = j  # standard deviation
        mean = 0  # mean of the gaussian distribution
        color = ["blue", "green", "red"]  # color of the noise - used for the file data
        noise_mat = np.random.normal(mean, STD, (image_input.shape[0], image_input.shape[1])).astype(np.uint8)   # create a matrix with random gaussian values
        # convert to 3D to add with the image
        noise_mask = np.zeros((image_input.shape[0], image_input.shape[1], 3), dtype=np.uint8)
        # convert to one noisy channel: choose the color by the list above
        noise_mask[:, :, i] = noise_mat
        noisy_image = cv.add(image_input, noise_mask)  # add the noise to the image
        save_path = "noisy_image.jpg"  # save the image by the path bellow
        cv.imwrite(save_path, noisy_image)
        # predict on an image
        detection_output = model.predict(source=save_path, conf=0.05, save=False)
        for r in detection_output:
            boxes = r.boxes
            for box in boxes:
                conf = box.conf.numpy()[0]  # get the probability of the bounding box
                name = int(box.cls.numpy()[0])  # get the class number of the bounding box
                if (name == 16) and (write_data is False):
                    # concatenate the data to write the desired information to file
                    data_2_file = "STD: " + str(STD) + " Channel: " + str(color[i]) + " probability: " + str(conf) + "\n"
                    # print(data_2_file)  # print the data to the console while save it to the file
                    with open("noise_data.txt", "a") as file:
                        file.write(data_2_file)
                        write_data = True
