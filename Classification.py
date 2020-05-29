'''
Sam Baker - 05/25
Practice with classification neural networks in Tensorflow
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import myModels

from tensorflow.keras import datasets, layers, models

def loadData():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels

def plotItem(item, images, labels):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.imshow(images[item], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels[item][0]])
    plt.show()



def main():
    train_images, train_labels, test_images, test_labels = loadData()

    #plotItem(1, train_images, train_labels)

    model = myModels.fccBigModel1()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=250,
                validation_data=(test_images, test_labels))



    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('test loss: ', test_loss, ', test accuracy: ', test_acc)

main()
