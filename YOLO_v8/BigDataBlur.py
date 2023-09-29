"""
the following code is used to detect a big data of images
its detect the images and save the data to a file by name, confidence and blur level by psi (physical coefficient)
in this case, the images are in format: c/d_b/g/r_1-2/0-10.jpg
the images are in the folder "inference/images/data_all"
the output is in the file "p_data.txt"
"""
import cv2 as cv
from ultralytics import YOLO

l1 = ["d"]  # d: dog - loop path string
l2 = ["b", "g", "r"]  # b: black, g: green, r: red - loop path string
l3 = [1, 2]  # 1-10 - loop path numbers
l4 = [0, 2, 4, 6, 8, 10]  # compression ratio loop
# loop through the lists to build the path to the image
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:
                path = "C:/Itzhak/BSc/4yr/Final_Project/Neural Network results/Blur/" + i + "_" + j + "_" + str(k) + "_" + str(m) + ".jpg"  # build the path to the image by lists
                image_input = cv.imread(path)  # read the image by path
                # load a pretrained YOLOv8n model
                model = YOLO("yolov8n.pt", "v8")

                # predict the blured image
                detection_output = model.predict(image_input, conf=0.05, save=False)

                # Loop through the detection output to save the data to a file by name and confidence
                for r in detection_output:
                    boxes = r.boxes
                    for box in boxes:
                        conf = box.conf.numpy()[0]  # get the probability of the bounding box
                        name = int(box.cls.numpy()[0])  # get the class number of the bounding box

                        # if the predicted is P( dog | dog ) save the data to the file by name, confidence and compression ratio
                        if name == 16:
                            if i == "d":
                                data_2_file = i + "_" + j + "_" + str(k) + "_" + str(m) + " " + str(conf) + "\n"
                                with open("Blur_data.txt", "a") as file:
                                    file.write(data_2_file)
