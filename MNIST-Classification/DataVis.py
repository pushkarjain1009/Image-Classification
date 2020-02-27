import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#Dataset Related Work

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirts', 'Trousers', 'pullovers', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel_Boot' ]

print("Train Image Dim: ", train_images.shape)
print("Train Image Dim: ", len(train_labels))

print("Test Image Dim: ", test_images.shape)
print("Test Image Dim: ", len(test_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


