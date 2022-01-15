from evaluator import FaceShapeEvaluator

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
import tensorflow

app = Flask(__name__)

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_virtual_device_configuration(gpus[0], [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])


face_classifier = FaceShapeEvaluator('/root/servers/server_1/model_ver2.h5')


@app.route('/getmethod', methods=['GET'])
def sample():
    return 'hi'


@app.route('/face_classifier', methods=['POST'])
def uploader_file():
    if request.method == 'POST':
        try:
            f = request.files['img_file']
        except:
            return 'File is missing', 404

        f.save(secure_filename(f.filename))
        path = os.path.realpath(f.filename)
        result = face_classifier.evaluate(path)
        os.remove(path)

        return result



@app.route('/userLogin', methods = ['POST'])
def userLogin():
    user = request.get_json()#json 데이터를 받아옴
    return jsonify(user)# 받아온 데이터를 다시 전송

if __name__ == '__main__':
    app.run('0.0.0.0', port = 80, debug=True)
