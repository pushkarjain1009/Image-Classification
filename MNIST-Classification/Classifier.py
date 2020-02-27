import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from plot import *


#Dataset Related Work

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

np.save("train_images", np.array(train_images))
np.save("train_labels", np.array(train_labels))
np.save("test_images", np.array(test_images))
np.save("test_labels", np.array(test_labels))


class_names = ['T-shirts', 'Trousers', 'pullovers', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel_Boot' ]

#Classification Model

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax),
])

model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

model.fit(train_images, train_labels, epochs = 10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Model Accuracy: ", test_acc*100 ,"%")

predictions = model.predict(test_images)

np.save("predictions", np.array(predictions))

