import numpy as np
from cv2 import cv2
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def edgeDetection(img):
    M, N = np.shape(img)[0], np.shape(img)[1]
    imgSol = np.zeros((M,N)) #initialize

    #mask 3x3
    umbral = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    for i in range(1, M - 1):
        for j in range (1, N - 1):
            imTemp = img[i - 1 : i + 2, j - 1 : j + 2]
            temp = (umbral * imTemp).sum() #pass the filter

            if temp > 255:
                imgSol[i - 1][j - 1] = 1 #stronger edges
            else:
                imgSol[i - 1][j - 1] = 0

    return imgSol


nameField = "perfil"
extension = "jpg"
path = "./"
outputpath = ""

if not os.path.exists("perfil.jpg"):
    print('file does not exist')
    exit(0)

img = cv2.imread("perfil.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("src/outputByN" + nameField + "." + extension, img)
imgEdge = edgeDetection(img)
cv2.imwrite("fin.jpg", imgEdge)