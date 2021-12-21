import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import os 



def brightness(img):
    gamma = get_gamma(img)
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    return res

def get_gamma(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(gray, 256, [0, 256])
    
    v = int(len(hist) / 2)
    f = np.sum(hist[:v] >= 1000)
    g = np.sum(hist[v:] >= 1000)

    f2 = np.sum(hist[:v] == 0)
    g2 = np.sum(hist[v:] == 0)
    
    gamma = 1
    if(abs(f- g) > 10 or abs(f2-g2) > 10):
        if (abs(f2-g2) <= 5 and (f2 < 50 and g2 < 50)) and (abs(f-g) > 10):
            if f < g:
                gamma = (0.4*g + 137.6)/118
            elif f > g:
                gamma = (-0.55*f + 145.2)/111
                #gamma =  abs((f*0.6)/128 - 0.6) + 0.6

        elif f2 > g2:
            gamma = (0.7*f2 + 76.6)/108
            #gamma = (34.8 + 0.6*f2)/72
        else:
            gamma = (-0.4*g2 + 128)/104
            #gamma = abs((g2*0.6)/100 - 0.6) + 0.6
            #gamma = (g2*0.8)/30 - 0.8
    return abs(gamma)