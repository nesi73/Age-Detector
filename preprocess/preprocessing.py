import os
from cv2 import cv2
import numpy as np
from raiseBrigthness import brightness
from dca import Dca
from pandas import DataFrame
from multiprocessing import Pool
import multiprocessing as mp

num_cpu = mp.cpu_count()
print(num_cpu)

folder_database = "../databaseAgeMio/"
out_folder_database = "databaseAgeMioPreprocess/"
folder = os.listdir(folder_database)


def preprocess(file):
    try:
        
        image = cv2.imread(folder_database + file)
        file_preprocess = brightness(image)
        cv2.imwrite(folder_database + file, file_preprocess)

        dca = Dca(folder_database + file)
        cropped_faces = dca.get_cropped_faces()

        for i in range(len(cropped_faces)):
            output_split = file.split(".")
            output_file_name = output_split[0] + "_" + str(i) + "." + output_split[1]

            age = "adult"

            if int(file.split("_")[2]) < 18:
                age = "younger"

            cv2.imwrite(out_folder_database + output_file_name, cropped_faces[i])

            return [out_folder_database + output_file_name, age]
    except Exception as err:
        print("Error:" + str(err))

if __name__ == '__main__':
    df_list = []

    with Pool(num_cpu) as p:
        df_list = p.map(preprocess, folder)
