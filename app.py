from evaluator import FaceShapeEvaluator

from flask import Flask, jsonify, request, json
from werkzeug.utils import secure_filename
import os
import tensorflow
from flask_cors import CORS

import time
import subprocess

from keras.models import model_from_json
import librosa
import numpy as np
import pandas as pd


app = Flask(__name__)
CORS(app, resources={r'*': {'origins': '*'}})

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_virtual_device_configuration(gpus[0], [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])


face_classifier = FaceShapeEvaluator('/root/server/model_ver2.h5')


@app.route('/getmethod', methods=['POST'])
def sample():
    return 'hi'

@app.route('/time')
def get_current_time():
    return {'time': time.time()}

@app.route('/testMethod', methods=['POST'])
def create():
    print(request.is_json)
    params = request.get_json()
    print(params['user'])
    return 'ok'


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
        # response = app.response_class(
        #     response=json.dumps(result),
        #     status=200,
        #     mimetype='application/json'
        # )

        # return response
        # return jsonify({"face":result})
        return result
        # return result


@app.route('/voice_analyzer', methods=['POST', 'GET'])
def uploader_voice_file():
    if request.method == 'POST':
        try:
            f = request.files['voice_file']
        except:
            return 'File is missing', 404

        f.save(secure_filename(f.filename))
        path = os.path.realpath(f.filename)

        mysp=__import__("my-voice-analysis")
        result = mysp.mysptotal(f.filename, path)
        print(result)
        # command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(f.filename, os.path.join('/root/server/', 'changed.wav'))
        # subprocess.call(command, shell=True)
        # path = os.path.realpath("changed.wav")

        # result = mysp.myspsr("changed.wav", path)

        # os.remove(path)

    return str(result)


@app.route('/emotion_analyzer', methods = ['POST'])
def voice():
    if request.method == 'POST':
        try:
            f = request.files['voice_file']
        except:
            return 'File is missing', 404

        f.save(secure_filename(f.filename))
        path = os.path.realpath(f.filename)

        json_file = open('voice_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("Emotion_Voice_Detection_Model.h5")
        print("Loaded model from disk")

        data, sampling_rate = librosa.load(f.filename)
        X, sample_rate = librosa.load(f.filename, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        featurelive = mfccs
        livedf2 = featurelive

        livedf2= pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim= np.expand_dims(livedf2, axis=2)
        livepreds = loaded_model.predict(twodim, 
                                batch_size=32, 
                                verbose=1)
        prediction=livepreds.argmax(axis=1)
        print(livepreds)

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

        os.remove(path)

        return str(prediction)


if __name__ == '__main__':
    app.run('0.0.0.0', port = 80, debug=True)

