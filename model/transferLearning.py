import csv
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.utils import plot_model

from raiseBrigthness import brightness
from get_data import LoadData
from save_in_csv import Save
from paint_graphics import Paint

DROP_OUT = 0.35000000000000003
LEARNING_RATE = 0.002
MOMENTUM = 0.2753813367505939


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0


def train_model():
    # video_path = "video.mp4"
    # sample_video = load_video(video_path)[:100]
    # print(sample_video.shape)

    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/classification/4", output_shape=[1001]),
        tf.keras.layers.Dense(3, activation='softmax')
        #tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    """
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])"""


    data = LoadData()

    train_y = tf.convert_to_tensor(data.train_y, dtype=tf.float32)
    train_X = tf.convert_to_tensor(data.train_X, dtype=tf.float32)
    validation_X = tf.convert_to_tensor(data.validation_X, dtype=tf.float32)
    validation_y = tf.convert_to_tensor(data.validation_y, dtype=tf.float32)

    print(train_y.shape)
    print(train_X.shape)
    print(validation_X.shape)
    print(validation_y.shape)

    h = model.fit(train_X, train_y, epochs=50, batch_size=64)

    # h = model.fit_generator((train_X,train_y), steps_per_epoch=len(data.train_X) // 2, epochs=1)

    # model.save("redPrueba.h5")
    # export_saved_model(model, 'path_to_my_model.h5')
    model.save('path_to_my_model.h5', save_format="tf")
    # Model predictions
    predictions = model.predict(data.test_X)

    # Save the predictions in csv
    Save(data.test_y, predictions)


def normalize(array):
    print(array)
    return array / 255.0


def load_model():
    model = tf.keras.models.load_model('modelo2.h5', custom_objects={'KerasLayer': hub.KerasLayer})


    data = LoadData()
    
      
    results = model.evaluate(data.test_X, data.test_y)
    
    predictions = model.predict(data.test_X)
    Save(data.test_y, predictions)


load_model()
