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
NMOD_HUMAN = os.path.expanduser(config.nmod_human_path)
IMAGE_SIZE = (224, 224)


class Dca:
    def __init__(self, file):
        self.cropped_faces = []
        self.file = file
        rects, detector = self.face_detection()
        if len(rects) != 0:
            self.landmarks_detection(rects, detector)
        else:
            rects = self.face_detection_with_cnn()

            if len(rects) != 0:
                self.crop_cnn_img(rects)

    def crop_cnn_img(self, rects):
        image = cv2.imread(self.file)

        for (i, face) in enumerate(rects):
            crop = image[face.rect.top():face.rect.bottom(), face.rect.left():face.rect.right()]

            crop = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

            self.cropped_faces.append(crop)

    def face_detection_with_cnn(self):
        cnn_face_detector = dlib.cnn_face_detection_model_v1(NMOD_HUMAN)

        image = cv2.imread(self.file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cnn_face_detector(gray, 1)

    def face_detection(self):
        if not os.path.exists(self.file):
            raise Exception("Sorry, file does not exist")
        else:
            detector = dlib.get_frontal_face_detector()

            image = cv2.imread(self.file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            return detector(gray, 1), detector

    def face_remap(self, shape_input):
        remapped_image = cv2.convexHull(shape_input)
        return remapped_image

    def landmarks_detection(self, rects, detector):
        image = cv2.imread(self.file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out_face = np.zeros_like(image)
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

        cont = 0
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)  # Create landmark object
            shape = face_utils.shape_to_np(shape)

            aligned_face = align_face_dlib(image, shape)

            aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            aligned_rects = detector(aligned_gray, 1)

            for (j, aligned_rect) in enumerate(aligned_rects):
                aligned_shape = predictor(aligned_gray, aligned_rect)
                aligned_shape = face_utils.shape_to_np(aligned_shape)

                p = self.icosagon(aligned_shape)

                m = 0
                for l in range(18, 27 + 1):
                    aligned_shape[l] = p[m]
                    m += 1

                counter = 0
                for item in aligned_shape:
                    aligned_shape[counter][0] = aligned_shape[counter][0] if aligned_shape[counter][0] > 0 else 0
                    aligned_shape[counter][1] = aligned_shape[counter][1] if aligned_shape[counter][1] > 0 else 0

                    counter += 1

                remapped_shape = self.face_remap(aligned_shape)

                c = remapped_shape[0:27]
                feature_mask = np.zeros((aligned_face.shape[0], aligned_face.shape[1]))
                cv2.fillConvexPoly(feature_mask, c, 1)

                ext_left, ext_right, ext_top, ext_bot = self.crop_face(c)

                feature_mask = feature_mask.astype(np.bool)
                out_face[feature_mask] = aligned_face[feature_mask]
                crop = out_face[ext_top:ext_bot, ext_left:ext_right]

                crop = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

                if len(rects) > cont:
                    self.cropped_faces.append(crop)

                cont += 1

    def icosagon(self, aligned_shape):
        p1 = aligned_shape[15]
        p2 = aligned_shape[34]

        distance_1 = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

        p1 = aligned_shape[3]
        p2 = aligned_shape[34]

        distance_2 = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

        avg_distance = (distance_1 + distance_2) / 2

        center_point_x = aligned_shape[34][0]
        center_point_y = aligned_shape[34][1] - int(avg_distance)

        N = 20

        p = np.zeros((11, 2))
        for k in range(0, N):
            if k >= 11:
                break

            p[k][0] = int(center_point_x - int(avg_distance) * math.cos(k * 2 * math.pi / N))
            p[k][1] = int(center_point_y - int(avg_distance) * math.sin(k * 2 * math.pi / N))

            p[k][0] = p[k][0] if p[k][0] > 0 else 0
            p[k][1] = p[k][1] if p[k][1] > 0 else 0

        return p

    def crop_face(self, c):
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot = tuple(c[c[:, :, 1].argmax()][0])

        cutTop = ext_top[1]
        cutBot = ext_bot[1]

        cutLeft = ext_left[0]
        cutRight = ext_right[0]

        imgH = cutBot - cutTop
        imgW = cutRight - cutLeft

        diffSide = (imgH - imgW) / 2

        cutLeft -= diffSide

        if cutLeft < 0:
            cutRight += abs(cutLeft)
            cutLeft = 0

        cutRight += diffSide


        if not isinstance(cutRight,int): #cutRight.is_integer():
            cutRight = math.floor(cutRight)

        if not isinstance(cutLeft,int): #cutLeft.is_integer():
            cutLeft = math.floor(cutLeft)

        imgW = cutRight - cutLeft

        if imgW != imgH:
            print('Error w,h', imgW, imgH)

        cutLeft = int(cutLeft)
        cutRight = int(cutRight)

        return cutLeft, cutRight, cutTop, cutBot

    def get_cropped_faces(self):
        return self.cropped_faces
