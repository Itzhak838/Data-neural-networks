"""
This script design to adding a square noise to images
 and then predicts on it using a pretrained YOLOv8n model.
 The target is to check the resolution limit of the data.
"""
import cv2 as cv
from ultralytics import YOLO

l1 = ["d"]  # d: dog - loop path string - dog
l2 = ["b", "g", "r"]  # b: black, g: green, r: red - loop path string
l3 = list(range(1, 11))  # 1-10 - loop path numbers
l4 = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]  # n*n black square
for i in l1:
    for j in l2:
        for k in l3:
            for m in l4:
                write_data = False  # check if the specific data was written to the file
                path = "inference/images/data_all/" + i + "_" + j + "_" + str(k) + ".jpg"  # build the path to the images by for loop
                img = cv.imread(path)  # read the image by path
                # make the square in the center of the image
                x_center = img.shape[0] // 2
                y_center = img.shape[1] // 2
                # adding a stop condition when the square is out of the image
                if (x_center - m >= 0) and (y_center - m >= 0) and (x_center + m//2 <= img.shape[0]) and (y_center + m//2 <= img.shape[1]):
                    img[x_center - m // 2:x_center + m // 2, y_center - m // 2:y_center + m // 2] = 0, 0, 0  # add the square to the image-black square

                    # Save the noisy image
                    save_path = 'noisy_image.jpg'
                    cv.imwrite(save_path, img)

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
                                data_2_file = i + "_" + j + "_" + str(k) + " Dimension of noise: " + str(m) + '*' + str(m) + " probability: " + str(conf) + "\n"
                                print(data_2_file)
                                with open("noise_square.txt", "a") as file:
                                    file.write(data_2_file)
                                    write_data = True
