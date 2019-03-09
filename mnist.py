import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from keras.models import load_model
img_rows, img_cols = 28, 28

#---Load----------
myload_model=load_model('train_m_a.h5py')
#----------------------------------------------------------------------
image = cv2.imread('vt_11.jpg')
im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray,(3,3),0)
cv2.imwrite("image_gauss_6.jpg",im_blur)
im,thre = cv2.threshold(im_blur,150,255,cv2.THRESH_BINARY_INV)
_,contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#rects = [cv2.boundingRect(cnt) for cnt in contours]
cv2.imwrite("image_theshold_6.jpg",thre)

#---------------
for i in contours:
    (x,y,w,h) = cv2.boundingRect(i)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    roi = thre[y:y+h,x:x+w]
    cv2.imwrite('image_rect_6.jpg',roi)
    roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
    cv2.imwrite('image_pad_6.jpg',roi)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    #-------------------------------
    roi =roi.reshape(1,img_rows, img_cols, 1)
    nbr= myload_model.predict(roi)
    idx = np.argmax(nbr)
    print(idx)
    cv2.putText(image, str(idx), (x, y),cv2.FONT_HERSHEY_DUPLEX,1.5, (0, 0, 255),2)
    cv2.imshow("image",image)
    
cv2.imwrite("image_pand_30.jpg",image)
cv2.waitKey()
cv2.destroyAllWindows()