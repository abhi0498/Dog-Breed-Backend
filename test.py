from PIL import Image
from flask_cors import CORS, cross_origin
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from model import load_img, predict
from flask import Flask, render_template, request, jsonify, json
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# img = Image.open('./static/images/caputre.jpg')
# img = img_to_array(img)[:, :, :3]
# img = cv2.resize(img, (224, 224))
# print(img.shape)

# base_model = Xception(weights='imagenet', include_top=False)
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(120, activation='softmax')(x)


# for layer in base_model.layers:
#     layer.trainable = False

# xception = Model(inputs=base_model.input, outputs=predictions)
# xception.load_weights('./models/dog_class.h5')

# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# xception.compile(optimizer=optimizer,
#                  loss='categorical_crossentropy', metrics=['accuracy'])


# img = np.expand_dims(img, axis=0)
# p = xception.predict(img)


dog = load_model('./models/dogornot (2).hdf5')
DOgOrNot = {'airplane': 0,
            'car': 1,
            'cat': 2,
            'dog': 3,
            'flower': 4,
            'fruit': 5,
            'motorbike': 6,
            'person': 7}

DOgOrNot = {DOgOrNot[k]: k for k in DOgOrNot}

img = Image.open('./static/images/7-faceforward.jpg')
img = np.array(img)
img = img[:, :, :3]
img = cv2.resize(img, (224, 224))
img = preprocess_input(img)
img = np.array([img])
pred = dog.predict(img)

print(np.argsort(-pred))
print(np.sort(pred)[0][-1]*100)
