import tensorflow as tf
from tensorflow import keras
import ssl
import urllib.request

# Load the MNIST digits dataset.
ssl._create_default_https_context = ssl._create_unverified_context
mnist = keras.datasets.mnist

# Split the dataset into training and testing sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model architecture.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

# Choose an optimizer and loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model.
model.fit(x_train, y_train, epochs=10)

# Evaluate the model on the test data using `evaluate`
print("\nEvaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=64)
print("test loss, test accuracy:", results)
