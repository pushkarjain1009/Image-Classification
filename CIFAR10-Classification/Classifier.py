from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

class_names = ["Aeroplain", "Automobiles", "Bird", "Cat", "Deer", 
               "Dog", "Frog", "Horse", "Ship", "Truck"]


def conv_net():
    
    model = Sequential()
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape = [32, 32, 3]))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same', activation='relu'))
    
    model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(units = 128, activation='relu'))
    model.add(Dense(units = 256, activation='relu'))
    model.add(Dense(units = 512, activation='relu'))
    model.add(Dense(units = 1024, activation='relu'))
    
    model.add(Dense(units=10, activation='softmax'))

    return model

model = conv_net()
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

training = model.fit(train_images, train_labels, batch_size=10, epochs=20, verbose=1, validation_data=(test_images, test_labels))

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(test_images)

mat = confusion_matrix(test_labels, y_pred)

plot_confusion_matrix(mat,figsize=(9,9))
