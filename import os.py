import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1) 
# x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=10)

# model.save('my_model1.keras')

model = tf.keras.models.load_model('my_model.keras')

# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

img = cv2.imread("tests/test.png", cv2.IMREAD_GRAYSCALE)  # Read in grayscale
img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
img = img / 255.0  # Normalize the image to [0, 1]
img = np.expand_dims(img, axis=-1)  # Add the channel dimension
img = np.expand_dims(img, axis=0)  # Add the batch dimension

prediction = model.predict(img)
print(f"The number is: {np.argmax(prediction)}")
plt.imshow(img.squeeze(), cmap=plt.cm.binary)
plt.show()