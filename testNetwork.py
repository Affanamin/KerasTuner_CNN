#import install_requirements

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# FOR TESTING

mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

test_images = test_images / 255.0
test_images=test_images.reshape(len(test_images),28,28,1)

model = tf.keras.models.load_model('model.h5')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

prediction = model.predict(test_images)


(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
test_images = test_images / 255.0

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel("Actual: "+class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

