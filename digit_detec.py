from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import numpy as np
import cv2

#Setting some parameters
#batch_size = 128
#num_classes = 10
#epochs = 1
# input image dimensions
img_rows, img_cols = 28, 28
#---------------------------------------------
#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plot the first image in the dataset
#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
#--------------------------------------------
#load mnist dataset, split between train and test sets
#x_train, y_train), (x_test, y_test) = mnist.load_data()
#print('tap test 1:',x_test)
#print('size tap test 1:',x_test.shape)
#Reshape data
#if K.image_data_format() == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
#else:
#    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#    input_shape = (img_rows, img_cols, 1)

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#print('tap test 2:',x_test)
#print('size tap test 2:',x_test.shape)
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

myload_model=load_model('train_9977.h5py')
#myload_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = myload_model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#test
image = cv2.imread('mnist15.jpg',0)
output_2=image.copy()
image=image.reshape(1,img_rows,img_cols,1)

#image = image.astype("float") / 255.0
yFit = myload_model.predict(image)[0]
yFit1 = myload_model.predict(X_test[:4])
idx = np.argmax(yFit)
idx1 = np.argmax(yFit1)
print(idx)
print(yFit)
print(idx1)
print(yFit1)
cv2.putText(output_2,str(idx),(1,1),cv2.FONT_HERSHEY_SIMPLEX,1,(200,255,155),2,cv2.LINE_AA)
cv2.imshow('result',output_2)
print('accuracy:',yFit)
print('number:',idx,' with accuracy=',yFit[idx])
cv2.waitKey(0) 	
cv2. destroyAllWindows()