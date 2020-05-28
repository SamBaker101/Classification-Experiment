'''
Sam Baker - 05/25
Practice with classification neural networks in Tensorflow
Helper file for models
'''

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def model1():
    '''
    Example given at https://www.tensorflow.org/tutorials/images/cnn
    Used as benchmark

    5 Epochs: test loss:  0.8747206167221069 , test accuracy:  0.6974
    10 Epochs: test loss: 0.9241683521270752, test accuracy: 0.6956
    20 Epochs: test loss:  1.1655726622581482 , test accuracy:  0.6974
    '''
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    return model

def fcModel1():
    '''
    Shallow Fully Connected Layer - expecting very low results
    5 Epochs: test loss:  1.6842042057037354 , test accuracy:  0.401
    10 Epochs: test loss:  1.5978460956573486 , test accuracy:  0.4334
    20 Epochs: test loss:  1.5800182376861571 , test accuracy:  0.4386

    '''
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    model.add(layers.Dense(64, activation='sigmoid', ))
    model.add(layers.Dense(10))

    model.summary()

    return model

def fcModel2():
    '''
    Deeper Fully connected layer, 128, 64, 32
    5 Epochs: test loss:  1.775723888015747 , test accuracy:  0.3561
    10 Epochs: test loss:  1.6631084827423095 , test accuracy:  0.4016
    20 Epochs: test loss: 1.565459051513672 , test accuracy:  0.4395

    for the lols :
    50 Epochs: test loss:  1.5183490997314453 , test accuracy:  0.4616
    100 Epochs: test loss:  1.5355622863769531 , test accuracy:  0.4788
    '''
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    model.add(layers.Dense(128, activation='sigmoid', ))
    model.add(layers.Dense(64, activation='sigmoid', ))
    model.add(layers.Dense(32, activation='sigmoid', ))
    model.add(layers.Dense(10))

    model.summary()

    return model

def cnnModel1():
    '''
    Simple CNN model (one conv layer, 1 64 Node hidden layer
    5 Epochs: test loss:  1.0826689511299132 , test accuracy:  0.6222
    10 Epochs: test loss:  1.0612340730667114 , test accuracy:  0.6451
    20 Epochs: 1.1198993400573731 , test accuracy:  0.628 - accuracy has begun to drop

    '''

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    return model