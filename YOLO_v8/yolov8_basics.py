from ultralytics import YOLO

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  

path = "C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\red.jpeg"

# predict on an image
detection_output = model.predict(source=path, conf=0.5, save=True)

# Display a tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].np())
