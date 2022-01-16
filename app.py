from model.face_shape_classifier.face_evaluator import FaceShapeEvaluator
from model.voice_analyzer.voice_evaluator import VocieEvaluator

from flask import Flask, jsonify, request, json
from werkzeug.utils import secure_filename
import os
import tensorflow
from flask_cors import CORS

import subprocess

from keras.models import model_from_json
import librosa
import numpy as np
import pandas as pd


app = Flask(__name__)
CORS(app, resources={r'*': {'origins': '*'}})

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_virtual_device_configuration(gpus[0], [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])


face_classifier = FaceShapeEvaluator('/root/server/model/face_shape_classifier/model_ver2.h5')

voice_analyzer = VocieEvaluator()


@app.route('/face_classifier', methods=['POST'])
def uploader_img_file():
    if request.method == 'POST':
        try:
            f = request.files['img_file']
        except:
            return 'File is missing', 404

        f.save(secure_filename(f.filename))
        path = os.path.realpath(f.filename)
        result = face_classifier.evaluate(path)
        os.remove(path)

        print(result)

        return result


@app.route('/voice_analyzer', methods = ['POST'])
def voice():
    if request.method == 'POST':
        try:
            f = request.files['voice_file']
        except:
            return 'File is missing', 404

        f.save(secure_filename(f.filename))
        path_m4a = os.path.realpath(f.filename)

        wav_file = 'wav_voice_file.wav'
        command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(f.filename, os.path.join('/root/server/', wav_file))
        subprocess.call(command, shell=True)
        path_wav = os.path.realpath(wav_file)

        result = voice_analyzer.evaluate(wav_file)

        #label-list
        # 0 - female_angry
        # 1 - female_calm
        # 2 - female_fearful
        # 3 - female_happy
        # 4 - female_sad
        # 5 - male_angry
        # 6 - male_calm
        # 7 - male_fearful
        # 8 - male_happy
        # 9 - male_sad

        os.remove(path_m4a)
        os.remove(path_wav)

        return result


if __name__ == '__main__':
    app.run('0.0.0.0', port = 80, debug=True)

