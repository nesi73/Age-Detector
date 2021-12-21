import numpy as np
from cv2 import cv2
import os
from matplotlib import pyplot as plt

"""
median filter is used to reduce noise in images. The filtering algorithm will scan the entire image, 
using a small matrix (like the 3x3 depicted above), and recalculate the value of the center pixel by 
simply taking the median of all of the values inside the matrix.
"""

def filterMendian(img):
    M, N = np.shape(img)
    container = np.copy(img)

    for i in range (1, M - 1): #Limits matrix
        for j in range (1, N - 1):
            m = [img[i-1, j - 1], img[i - 1, j], img[i - 1, j + 1], img[i, j - 1], img[i, j], img[i, j + 1], img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]]
            m.sort() #sort array
            container[i][j] = m[int(len(m) / 2)] #middle element

    return container

def main():
    folder_database = "preprocess/"
    out_folder_database = "databaseAge/"
    folder = os.listdir(folder_database)

    for file in folder:
        image = cv2.imread(folder_database + file)
        im = filterMendian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        cv2.imwrite("mediana/"+file, im)
    # imgMedian = filterMendian(img)
    # cv2.imwrite("src/outputByN/" + nameFile, img)
    # imgMedian = cv2.medianBlur(img, 1)
    # processFiles.writeFile(outputpath,nameFile, imgMedian)

main()