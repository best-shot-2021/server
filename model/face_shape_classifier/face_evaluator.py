import tensorflow as tf 
import numpy as np
from keras.preprocessing import image

class FaceShapeEvaluator():
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path)

    def evaluate(self, test_data):
        test_image = image.load_img(test_data, target_size = (200,200))
        #test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = self.model.predict(test_image)

        if prediction[0][1] == 1:
            result = "heart"
        elif prediction[0][0] == 1:
            result = "oblong"
        elif prediction[0][2] == 1:
            result = "oval"
        elif prediction[0][3] == 1:
            result = "round"
        elif prediction[0][4] == 1:
            result = "square"

        print(result)

        if(result=="square"):
            final_result = str(0)
        elif(result=="heart" or result=="round"):
            final_result = str(1)
        elif(result=="oval" or result=="oblong"):
            final_result = str(2)

        return final_result
