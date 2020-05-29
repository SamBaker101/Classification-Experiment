'''
Sam Baker - 05/25
Practice with classification neural networks in Tensorflow
Helper file for models

Current Best Models:
5 Epochs: cnnDropModel4: test loss:  0.8464962710380555 , test accuracy:  0.7083
10 Epochs: cnnDropModel4: test loss:  0.7574391759395599 , test accuracy:  0.7431
20 Epochs: cnnDropModel4: test loss:  0.7643481120109558 , test accuracy:  0.751

Overall:
30 Epochs: cnnDropModel4: test loss: 0.8349342082977295 , test accuracy:  0.7549
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

def fccBigModel1():
    '''
    Large fully connected network
    I expect extremely low results for 5, 10, 20 but will attempt very large training session to see what happens

    5 Epochs: test loss:  2.0169111042022707 , test accuracy:  0.2202
    10 Epochs: test loss:  1.8814141105651856 , test accuracy:  0.2967
    20 Epochs: test loss:  1.7632045572280883 , test accuracy:  0.3611

    50 Epochs: test loss: 1.569729489326477 , test accuracy:  0.4341
    100 Epochs: test loss:  1.5932472543716432 , test accuracy:  0.4685
    250 Epochs: test loss:  3.286265371322632 , test accuracy:  0.4325

    Overfitting became a serious problem in longer training sessions and
    the training time is excessive for such a straight-forward problem

    Likely requires far more training data to be effective
    '''

    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    model.add(layers.Dense(128, activation='sigmoid', ))
    model.add(layers.Dense(256, activation='sigmoid', ))
    model.add(layers.Dense(512, activation='sigmoid', ))
    model.add(layers.Dense(256, activation='sigmoid', ))
    model.add(layers.Dense(128, activation='sigmoid', ))
    model.add(layers.Dense(64, activation='sigmoid', ))
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

def cnnModel2():
    '''
    Larger conv network with 2 fc layers
    First model to outperform benchmark (model 1) on 5 and 10 epochs
    Issues with overfit at 20 epochs (though still outperforming model 1)

    5 Epochs: test loss: 0.8817, test accuracy: 0.6914
    10 Epochs:test loss:  0.9037117343902588 , test accuracy:  0.7179
    20 Epochs: 1.2556573266029358 , test accuracy:  0.7126

    '''

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    return model

def cnnDropModel1():
    '''
    cnnModel2 with dropout layer added after Conv layers to combat over-fitting

    Much slower training (as expected) significantly lower results for 5,10 and 20 Epochs
    accuracy is however steadily increasing

    5 Epochs: test loss: 1.3817854423522948 , test accuracy:  0.5264
    10 Epochs: test loss: 1.179222657585144 , test accuracy:  0.582
    20 Epochs: test loss:  0.9782691495895386 , test accuracy:  0.6643

    Attempting longer training sessions to see if accuracy continues to improve:

    30 Epochs: test loss:  0.900225507068634 , test accuracy:  0.6897
    40 Epochs: test loss: 0.8876133846282959 , test accuracy:  0.6914

    100 Epochs: test loss:  0.8299943448066711 , test accuracy:  0.7197

    At 100 Epochs accuracy is still increasing each Epoch however the training time is getting
    beyond the current scope of this project. In my next cnnDropModel I will drop the drop rate
    to attempt to compensate
    '''

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    return model

def cnnDropModel2():
    '''
    Same build as cnnDropModel1 except with lower drop rate 0.1 to attempt to balance long
    training times

    Training significantly faster, outperforms cnnDropModel1 after 5 epochs (very close to model1 accuracy)

    New best result at 10 Epochs, 20 Epochs and overall

    5 Epochs: test loss: test loss:  0.8833148361206055 , test accuracy:  0.6924
    10 Epochs: test loss: test loss:  0.8093138602256775 , test accuracy:  0.7218
    20 Epochs: test loss: test loss:  0.7471956545829773 , test accuracy:  0.7484

    Attempting longer training sessions to see if accuracy continues to improve:

    30 Epochs: test loss:  0.7935745679855347 , test accuracy:  0.7538

    '''

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    return model

def cnnDropModel3():
    '''
    Experimenting with different drop rates through model

    Dropouts[0.1, 0.25, 0.5]
    5 Epochs: test loss: 0.9480008388519288 , test accuracy:  0.6733
    10 Epochs: test loss: 0.8264644397735595 , test accuracy:  0.7156
    20 Epochs: test loss: 0.7595288729667664 , test accuracy:  0.7402

    Dropouts [0.5, 0.25, 0.1]
    5 Epochs: test loss:  1.0232170785903931 , test accuracy:  0.6406
    10 Epochs: test loss:  0.8896497042655945 , test accuracy:  0.69
    20 Epochs: test loss: test loss:  0.8113080429077149 , test accuracy:  0.7198
    '''

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    return model

def cnnDropModel4():
    '''
    added another dense layer, expanded conv layer size

    New best for 5 epochs

    5 Epochs: test loss:  0.8464962710380555 , test accuracy:  0.7083
    10 Epochs: test loss:  0.7574391759395599 , test accuracy:  0.7431
    20 Epochs: test loss:  0.7643481120109558 , test accuracy:  0.751
    30 Epochs: test loss: 0.8349342082977295 , test accuracy:  0.7549

    '''

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(96, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    return model