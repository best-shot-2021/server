from .face_shape_evaluator import FaceShapeEvaluator

import io
from PIL import Image
import numpy as np
from flask import Flask                                            
from flask import request                                        
from flask import render_template, redirect, url_for, request    
from flask import jsonify                                        
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence                                
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/exam") # 접속 ip혹은 도메인 뒤 붙는 라우터 이름
def predict():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory("/이미지 경로/exam",target_size=(100,100), batch_size=100, class_mode='categorical')
    new_model = keras.models.load_model('모델이름.h5')
    new_model.summary()
    loss, acc = new_model.evaluate_generator(test_generator, steps=5) 
    data = {"success": False} # dictionary 형태의 데이터를 만들어 놓고 (딕셔너리에 데이터 넣는 방법1 : dictionary_name = {key:value}) 
    
    data["loss_accuracy"] = acc # 호출한 모델의 정확도를 넣습니다. (딕셔너리에 데이터 넣는 방법2 : dictionary_name[key] = value)
 
    data["success"] = True # 같은 방식으로 가지고 있는 key의 value를 바꿀수 있습니다.
            
    return jsonify(str(acc)) # '/exam'으로 요청을 보낸곳으로 값을 반환하는데에 json형태로 만들어 보내는데 jsonify를 하려면 데이터가 str 형태여야 합니다.

if __name__ == "__main__": # terminal에서 python 인터프리터로 .py 파일을 실행하면 무조건 이 부분을 찾아 실행합니다.
                           # C의 main
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host="0.0.0.0") # app.run을 해줘야 flask 서버가 구동됩니다. 
                            # host="0.0.0.0"은 외부에서 해당 서버 ip 주소 접근이 가능하도록 하는 옵션입니다.
