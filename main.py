from face_shape_evaluator import FaceShapeEvaluator
import flask
from flask import Flask, request, render_template
import joblib
import imageio
import numpy as np
import torch
import os
import tensorflow as tf

app = Flask(__name__)

# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        file = request.files['image']
        if not file: return render_template('index.html', label="No Files")

        # img = imageio.imread(file)
        # img = img[:, :, :3]
        # img = img.reshape(1, -1)

        img = file

        # 입력 받은 이미지 예측
        # prediction = model.evaluate(img)

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        # label = str(np.squeeze(prediction))

        # 숫자가 10일 경우 0으로 처리
        # if label == '10': label = '0'

        # 결과 리턴
        return render_template('index.html', label=1)


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    # model = joblib.load('./model/model.pkl')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''    
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    # model = FaceShapeEvaluator('/root/server/model.h5')
    print('Model launched')
    app.run(host='0.0.0.0', debug=True)
