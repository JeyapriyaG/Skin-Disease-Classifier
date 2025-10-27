import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ensure models folder exists
os.makedirs("models", exist_ok=True)

# simple dummy model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Flatten(),
    layers.Dense(7, activation='softmax')  # number of classes in your app
])

# save to expected location
model.save("models/final_model.h5")
print("Dummy model created at models/final_model.h5")
