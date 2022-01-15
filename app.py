from evaluator import FaceShapeEvaluator

from flask import Flask, jsonify, request, json
from werkzeug.utils import secure_filename
import os
import tensorflow
from flask_cors import CORS

import time


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
        result = mysp.myspsyl(f.filename, path)
        os.remove(path)

    return str(result)


@app.route('/userLogin', methods = ['POST'])
def userLogin():
    user = request.get_json()#json 데이터를 받아옴
    return jsonify(user)# 받아온 데이터를 다시 전송

if __name__ == '__main__':
    app.run('0.0.0.0', port = 80, debug=True)

