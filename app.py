from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception

from flask import Flask, render_template, request, jsonify, json
from model import load_img, predict
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import RMSprop
from flask_cors import CORS, cross_origin
from model import dogornot
import cv2
import numpy as np
from PIL import Image
import os
app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(120, activation='softmax')(x)


for layer in base_model.layers:
    layer.trainable = False

xception = Model(inputs=base_model.input, outputs=predictions)
xception.load_weights('./models/dog_class.h5')

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

xception.compile(optimizer=optimizer,
                 loss='categorical_crossentropy', metrics=['accuracy'])


@app.route('/')
@cross_origin()
def home():
    return app.send_static_file('index.html')


@app.route('/api', methods=['GET', 'POST'])
@cross_origin()
def api():

    print(request.files['image'])
    # read file from HTTP request
    if(request.files['image'].filename != ''):
        file = request.files['image']
        file_path = './static/images/' + file.filename

    # save image in a local directory
    if (file.filename in os.listdir('./static/images')):
        os.remove(file_path)
    file.save(file_path)
    local_file = Image.open(file_path)

    # preprocess image
    img = np.array(local_file)
    img = img[:, :, :3]
    img = cv2.resize(img, (224, 224))
    pred = dogornot(img)
    if (pred != 'dog'):
        return jsonify({'pred': pred, 'isDog': False})
    print(img.shape)
    pred = predict(img, xception)
    print('[Hello]', pred)
    return jsonify({'pred': pred, 'isDog': True})
