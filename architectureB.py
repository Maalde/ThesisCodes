import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

#######################################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##########################################################
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#################################################################

import cv2
from PIL import Image
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.style.use('classic')
from sklearn.metrics import classification_report,confusion_matrix,multilabel_confusion_matrix

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,LSTM,Permute,Reshape,Activation, Dropout, Flatten, Dense, Bidirectional,GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten, Activation,Dropout,BatchNormalization
from tensorflow.python.util import deprecation
import warnings

warnings.simplefilter(action='ignore',category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False

# from tensorflow.keras import backend as K


###############################################################################################

from import_dataset import *
####################################################################################################

####MODEL ARCHITECTURE 2, CRNN

INPUT_SHAPE = (SIZE, SIZE, 3)
model = Sequential()
model.add(Conv2D(32, (7, 7), input_shape=INPUT_SHAPE,strides=(2,2),activation='relu')) #can change to 7*7 filter size
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

model.add(Conv2D(32, (3, 3), padding='same',strides=(2,2), kernel_initializer='he_uniform',activation='relu'))
model.add(Conv2D(32, (3, 3),strides=(2,2), kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), strides=(2,2), kernel_initializer='he_uniform',activation='relu'))
model.add(Conv2D(32, (3, 3),padding='same',strides=(2,2), kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3),padding='same',strides=(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(Conv2D(64, (3, 3),padding='same',strides=(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3),padding='same',strides=(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(Conv2D(64, (3, 3),padding='same',strides=(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3),padding='same', strides=(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(Conv2D(128, (3, 3),padding='same',strides=(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same',strides=(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(Conv2D(128, (3, 3),padding='same', strides=(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3),padding='same',strides=(2,2), kernel_initializer='he_uniform',activation='relu'))
model.add(Conv2D(256, (3, 3),padding='same',strides=(2,2), kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3),padding='same',strides=(2,2), kernel_initializer='he_uniform',activation='relu'))
model.add(Conv2D(256, (3, 3),padding='same',strides=(2,2), kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
model.add(BatchNormalization())

model.add(Reshape((256, -1)))
model.add(Permute((2, 1)))
#model.add(LSTM(512,return_sequences=True))
#model.add(LSTM(256))
model.add(Bidirectional(LSTM(512,return_sequences=True)))
model.add(Bidirectional(LSTM(256,return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(BatchNormalization())

#model.add(GlobalAveragePooling2D())
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
#      initial_learning_rate=1e-2,
#      decay_steps=10000,
#      decay_rate=0.9)
# optimizer = tensorflow.keras.optimizers.Adam(learning_rate=lr_schedule)
# # #
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', 'AUC','Recall','Precision'])

#print(model.summary())

my_callbacks = [tf.keras.callbacks.CSVLogger('csv/', separator=",",  append=True)]

history = model.fit(x = X_train, y = label_train, epochs = 100, batch_size= 128, verbose=1, callbacks = my_callbacks, validation_data=(X_valid,label_valid))

####################################################################################################

#SAVE MODEL

model.save('models/')

#######################################################################################################

#VALIDATION PREDICTION

# load model

model = load_model('models/')

###################################################################################################################

losss,acc, auc,recall,precision = model.evaluate(X_valid, label_valid)
print("Accuracy = ", (acc * 100.0), "%")


#############################################################################################################

#################################################################################################################


Y_pred = model.predict(dataset_valid, batch_size=128, verbose=1)
y_pred = Y_pred > 0.75

print(confusion_matrix(label_valid, y_pred))
print(classification_report(label_valid, y_pred))
print(multilabel_confusion_matrix(label_valid, y_pred))




#################################################################################################################


#######################################################################################################

##Testing model


loz,acc, auc,recall,precision = model.evaluate(dataset_test, label_test)
print("Accuracy = ", (acc * 100.0), "%")


#########################################################################################

###########################################################################

Y_pred = model.predict(dataset_test, batch_size=128, verbose=1)
y_pred = Y_pred > 0.75


print(confusion_matrix(label_test, y_pred))
print(classification_report(label_test, y_pred))
print(multilabel_confusion_matrix(label_test, y_pred))