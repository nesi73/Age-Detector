from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, AveragePooling2D, Flatten, Dropout, LSTM
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Input
from keras_preprocessing.image import ImageDataGenerator
import os, csv
import numpy as np
import cv2
from get_data import LoadData
from paint_graphics import Paint
from save_in_csv import Save
# lstm forget gate, input gate, out gate

# variables
CLASSES = 2  # mayores/menores
BATCH_SIZE = 64  # number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
UNIT = 256  # output shape of the tensor that is produced by the layer and that will be the input of the next layer.
DROP_OUT = 0.35000000000000003  # to prevent overfitting
LEARNING_RATE = 0.1  # hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
MOMENTUM = 0.9 # to accelerate the descent
IMAGE_SIZE = 224  # the size of the image


def create_model():
    # create a ResNet50 statement
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pooling="avg")

    prediction = Dense(units=3, kernel_initializer="he_normal", use_bias=False, activation='softmax',
                       name="pred_age")(base_model.layers[-1].output)

    model = Model(inputs=base_model.input, outputs=prediction)

    return model


def save_in_csv(test_gen, predictions):
    y_true = test_gen.classes
    y_pred = np.array([np.argmax(x) for x in predictions])

    with open('deepUAge.csv', 'w') as f1:
        writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )

        for i in range(0, len(y_pred)):
            writer.writerow([y_pred[i], y_true[i]])


model = create_model()

sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)

model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics='mae')

# Load dataset
data = LoadData()

print(data.train_X.shape)

# Model fit
h = model.fit(data.train_X,data.train_y, epochs=100, verbose=1, batch_size=BATCH_SIZE, validation_data = (data.validation_X, data.validation_y))
print(h.history.keys())


model.save('modelo3Mejorado.h5', save_format="tf")
Paint(h)
# validation_loss = np.min(h.history['val_loss'])
# scores = model.predict_generator(test_gen, test_gen.samples)
# print("Error = ", scores)
predictions = model.predict(data.test_X)
results = model.evaluate(data.test_X, data.test_y)
print("test loss, test acc:", results)
Save(data.test_y, predictions)

