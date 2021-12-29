import argparse
import cv2
import imutils 
import numpy as np  
import matplotlib.pyplot as plt
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input image")
#ap.add_argument("-o", "--output", required=True, help="path to output image")
args = vars(ap.parse_args())
img = cv2.imread(args["input"])
print(img.shape)
os.system("pause")
#cv2.imwrite(args["output"], img)

