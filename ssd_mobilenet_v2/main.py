"""
the following code is used to detect objects on image.
"""
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt
# import numpy as np

# Load a pre-trained SSD model from TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
model = hub.load(model_url)


# Load and preprocess an image
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


# Specify the path to your image
image_path_ = "C:\\Itzhak\\BSc\\4yr\\Final_Project\\pythonProject\\yolov8-silva\\inference\\images\\data_all\\c_b_5.jpg"
# Perform object detection on the image
boxes_, classes_, scores_ = detect_objects(image_path_)

# Display the image with bounding boxes
img = Image.open(image_path_)
plt.imshow(img)
ax = plt.gca()

# Threshold for displaying detected objects
confidence_threshold = 0.05

for box, cls, score in zip(boxes_, classes_, scores_):
    if score > confidence_threshold:
        y_min, x_min, y_max, x_max = box
        width, height = img.size  # Get width and height using .size
        y_min, x_min, y_max, x_max = (
            int(y_min * height),
            int(x_min * width),
            int(y_max * height),
            int(x_max * width),
        )
        label = f"Object: {cls}, Probability: {score:.2f}"  # Include class and probability
        print(label)  # Print the label and probability to the screen
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            linewidth=2.0,
            color="red",
        )
        ax.add_patch(rect)

class_names = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    # Add more class mappings using the actual COCO class names and IDs...
}

plt.axis("off")
plt.show()
