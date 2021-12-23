from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import shutil, os
from time import sleep 
import glob

carplate_img = cv2.imread("testdata/Train09.jpg")


def carplate_detect(image):
    carplate_overlay = image.copy()
    
    carplate_rects = cv2.CascadeClassifier('haar_carplate.xml')
    signs = carplate_rects.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in signs:
        cv2.rectangle(carplate_overlay, (x+15, y+15), (x+w-20, y+h-10), (255, 0 ,0 ), 5)
       
        return carplate_overlay

detected_carplate_img = carplate_detect(carplate_img)
cv2.imshow("ddd", detected_carplate_img)
cv2.waitKey(0)