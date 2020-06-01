from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import urllib.request
import pickle
from tensorflow.keras.applications.resnet50 import preprocess_input

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = load_model('./models/dogornot (2).hdf5')
DOgOrNot = {'airplane': 0,
            'car': 1,
            'cat': 2,
            'dog': 3,
            'flower': 4,
            'fruit': 5,
            'motorbike': 6,
            'person': 7}

DOgOrNot = {DOgOrNot[k]: k for k in DOgOrNot}


def dogornot(img):
    img = preprocess_input(img)
    img = np.array([img])
    return DOgOrNot[np.argsort(-(model.predict(img)))[0][0]]


def decode(pred):
    with open('./models/classes.pkl', 'rb') as c:
        classes = pickle.load(c)

    new_classes = {}
    for i in classes:
        temp = classes[i]
        new_classes[temp] = i
    sorted_indices = np.argsort(-pred)
    for n in new_classes:
        new_classes[n] = ' '.join([i.capitalize()
                                   for i in new_classes[n].split('-')[1:]]).replace('_', ' ')
        new_classes[n] = ' '.join([i.capitalize()
                                   for i in new_classes[n].split()])
    top3_indices = np.argsort(-pred)[0][:3]
    print([[new_classes[i], f'{round(pred[0][i]*100,2)}']
           for i in top3_indices])
    return [[new_classes[i], f'{round(pred[0][i]*100,2)}'] for i in top3_indices]


def predict(img, model):
    X = cv2.resize(img, (224, 224))

    X = np.expand_dims(X, axis=0)
    X = X/255
    pred = model.predict(X)
    return decode(pred)


def load_img(url):
    image = Image.open(urllib.request.urlopen(url))
    img = np.array(image)
    return img
