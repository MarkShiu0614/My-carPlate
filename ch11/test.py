import cv2  
import numpy as np  
import matplotlib.pyplot as plt


#讀取圖片
imagePath = 'Haar-Training_carPlate/training/positive/rawdata/'
imgname = 'Train01'
img_jpg = '.jpg'
img = cv2.imread(imagePath+imgname+img_jpg)

# resize_photo
img1 = img.copy()
MAX_WIDTH =1000
rows, cols = img.shape[:2]
if cols > MAX_WIDTH:
    change_rate = MAX_WIDTH / cols
    img1 = cv2.resize(img1,(MAX_WIDTH, int(rows * change_rate)), interpolation= cv2.INTER_AREA)

# GrayImage
GrayImage = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print("讀入lenna圖的shape為", GrayImage.shape)

#高斯平滑
Gaussian = cv2.GaussianBlur(GrayImage, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

#中值濾波
Median = cv2.medianBlur(Gaussian, 5)

cv2.imshow('Gray.bmp', GrayImage)
cv2.imshow('resize', img1)
cv2.imshow('gaussian', Gaussian)
cv2.imshow('median', Median)
cv2.waitKey(0)

cv2.imwrite(imgname+'.bmp', img1)