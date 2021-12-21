import os
from imutils import face_utils
import numpy as np
import dlib
from cv2 import cv2
from align import align_face_dlib
import argparse
import config
import math

SHAPE_PREDICTOR = os.path.expanduser(config.shape_predictor_path)

parser = argparse.ArgumentParser(description='DCA face cropping tool')
parser.add_argument("--f", default='mujer1.jpg', type=str, help="path of image")
parser.add_argument("--o", default='mujer1.jpg', type=str, help="output path of cropped image")
parser.add_argument("--p", default='mujer2.jpg', type=str, help="output path of cropped image")
args = parser.parse_args()


"""
MAIN CODE STARTS HERE
"""
# load the input image, resize it, and convert it to greyscale

file_path = args.f
output_path = args.o
output_path_pruebas = args.p

if not os.path.exists(file_path):
    print('file does not exist')
    exit(0)
else:
    image = cv2.imread(file_path)
    out_face = np.zeros_like(image)

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cnn_face_detector  = dlib.cnn_face_detection_model_v1(SHAPE_PREDICTOR)
    rectsCNN = cnn_face_detector(gray, 1)
    
    # predictor = dlib.shape_predictor(SHAPE_PREDICTOR) #Load predictor

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    rectsLandmarks = detector(gray, 1)
    # detect faces in greyscale


    """ if len(rects) == 0:
        print('no face detected') """

    cont = 0
    # loop over the face detections
    for (i, face) in enumerate(rectsCNN):
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right()
        h = face.rect.bottom()

        # draw box over face
        cv2.rectangle(image, (x,y), (w,h), (0,0,255), 2)
        # cv2.imwrite("perfilCuadrado1.jpg", image)

    for (i,face) in enumerate(rectsLandmarks):
        # draw box over face
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0,255,0), 1)

    cv2.imwrite("perfilCuadrado3.jpg", image)