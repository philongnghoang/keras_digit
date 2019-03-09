import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
import cv2
import numpy as np
img_rows, img_cols = 28, 28

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plot the first image in the dataset
#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
"""
#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)
model.save('train3.h5py')
"""
myload_model=load_model('train3.h5py')
#-----------------------------------------------------------------------------
#predict first 4 images in the test set

image = cv2.imread('so2.jpg')
#im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im_gray = cv2.imread('soo.jpg',0)
im_blur = cv2.GaussianBlur(im_gray,(3,3),0)
im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
_,contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(cnt) for cnt in contours]
#cv2.imshow('hinh',thre)

#---------------
for i in contours:
    (x,y,w,h) = cv2.boundingRect(i)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    roi = thre[y:y+h,x:x+w]
    roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    roi =roi.reshape(1,img_rows, img_cols, 1)
    roi = roi.astype("float") / 255.0
    # Calculate the HOG features
    #roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),block_norm="L2")
    nbr= myload_model.predict(roi)[0]
    #nbr=model.predict(roi)[0]
    idx = np.argmax(nbr)
    #print(idx)
    #print(str(int(nbr[0])))

    #nbr = model.predict(np.array([roi_hog_fd], np.float32))
    #cv2.putText(image, str(int(nbr[0])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    cv2.putText(image, str(idx), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 225, 155), 3)
    #cv2.putText(image,str(idx),(1,1),cv2.FONT_HERSHEY_SIMPLEX,1,(200,255,155),2,cv2.LINE_AA)
    cv2.imshow("image",image)

#print(model.predict(X_test[:4]))
print(myload_model.predict(X_test[:4]))
print(y_test[:4])    
cv2.imwrite("image_pand2.jpg",image)
cv2.waitKey()
cv2.destroyAllWindows()