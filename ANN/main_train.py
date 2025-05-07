# import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models

DATA_TRAIN = 'dataset/image'
DATA_TEST = 'dataset/test'

Xtrain = []
Ytrain = []

Xtest = []
Ytest = []

dict = {'traindenvau': [1, 0, 0, 0, 0], 'trainjennie': [0, 1, 0, 0, 0], 'trainlisa': [0, 0, 1, 0, 0], 'trainsontung': [0, 0, 0, 1, 0], 'traintruonggiang': [0, 0,0, 0, 1],       
        'testdenvau': [1, 0, 0, 0, 0], 'testjennie': [0, 1, 0, 0, 0], 'testlisa': [0, 0, 1, 0, 0], 'testsontung': [0, 0, 0, 1, 0], 'testtruonggiang': [0, 0,0, 0, 1]}

# for i in os.listdir(DATA_TRAIN):
#     path = os.path.join(DATA_TRAIN, i)
#     print(i)
#     for j in os.listdir(path):
#         file_path = os.path.join(path, j)
#         img = Image.open(file_path)
#         print(img)

def getData(dirData, lstData):
    for i in os.listdir(dirData):
        i_path = os.path.join(dirData, i)
        lst_filename_path = []
        for filename in os.listdir(i_path):
            filename_path = os.path.join(i_path, filename)
            label = os.path.basename(os.path.dirname(filename_path))
            img = Image.open(filename_path).resize((128, 128))
            img = np.array(img)
            lst_filename_path.append((img, dict[label]))
        lstData.extend(lst_filename_path)
    return lstData

Xtrain = getData(DATA_TRAIN, Xtrain)
Xtest = getData(DATA_TEST, Xtest)

# print(Xtrain[498])
# print(Xtest[81])
# print(len(Xtest))

model_training = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(5, activation='softmax') 
])

# model_training_first.summary()

model_training.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_training.fit(
    np.array([x[0] for _, x in enumerate(Xtrain)]),
    np.array([y[1] for _, y in enumerate(Xtrain)]),
    epochs=15
)

model_training.save('model.h5')
models_loaded = models.load_model('model.h5')

