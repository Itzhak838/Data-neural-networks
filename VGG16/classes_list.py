"""
The code designed to predict an image objects in the VGG16 model
"""
from keras.applications import VGG16
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

# Create a random array as a placeholder for an image
random_image = np.random.random((1, 224, 224, 3))

# Predict the probabilities across all output classes
yhat = model.predict(random_image)

# Decode the prediction and get the class names
label = decode_predictions(yhat)
label = label[0][0]

# Print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))

# Create a dummy prediction with equal probabilities for all classes
dummy_prediction = np.expand_dims(np.ones(1000), 0)

# Decode the prediction
decoded = decode_predictions(dummy_prediction)
print(decoded)
# Get the class names
class_names = [t[1] for t in decoded[0]]

# Print the class names
for name in class_names:
    print(name)