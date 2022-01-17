from model.face_shape_classifier.face_evaluator import FaceShapeEvaluator
from model.voice_analyzer.voice_evaluator import VocieEvaluator

from flask import Flask, jsonify, request, json
from werkzeug.utils import secure_filename
import os
import tensorflow
from flask_cors import CORS

import subprocess
import logging

app = Flask(__name__)
CORS(app)
logging.getLogger('flask_cors').level = logging.DEBUG
# CORS(app, resources={r'*': {'origins': '*'}})
# CORS(app, support_credentials=True)
# CORS(app, supports_credentials=True, origins="localhost:3000")
# cors = CORS(app, resource={
#     r"/*":{
#         "origins":"*"
#     }
# })


gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_virtual_device_configuration(gpus[0], [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])


face_classifier = FaceShapeEvaluator('/root/server/model/face_shape_classifier/model_ver2.h5')
voice_analyzer = VocieEvaluator()


# @app.after_request
# def after_request(response):
#   response.headers.add('Access-Control-Allow-Origin', '*')
#   response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#   response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#   return response

@app.route('/face_classifier', methods=['POST'])
def face_img_classifying():
    if request.method == 'POST':
        try:
            f = request.files['img_file']
        except:
            return 'File is missing', 404

        f.save(secure_filename(f.filename))
        path = os.path.realpath(f.filename)
        result = face_classifier.evaluate(path)
        print(result)

        os.remove(path)

        return str(result)


@app.route('/voice_analyzer', methods = ['POST'])
def voice_m4a_analyzing():
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
        print(result)

        os.remove(path_m4a)
        os.remove(path_wav)

        return str(result)


@app.route('/mood_finder', methods=['POST'])
def mood_finding():
    if request.method == 'POST':
        try:
            params = request.get_json()
        except:
            return 'Json is missing', 404

        face_result = int(params['face'])
        voice_result = int(params['voice'])
        print(face_result)
        print(voice_result)

        #지적
        if(face_result==0 and voice_result==0):
            result = str(0)
        #섹시
        elif(face_result==2 and voice_result==1):
            result = str(1)
        #청순
        elif(face_result==2 and voice_result==0):
            result = str(2)
        #조용귀염
        elif(face_result==1 and voice_result==0):
            result = str(3)
        #유학파
        elif(face_result==0 and voice_result==1):
            result = str(4)
        #활발귀염
        elif(face_result==1 and voice_result==1):
            result = str(5)

        print(result)

        return result

if __name__ == '__main__':
    app.run('0.0.0.0', port = 80, debug=True)

