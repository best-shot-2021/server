from keras.models import model_from_json
import librosa
import numpy as np
import pandas as pd

class VocieEvaluator():
    def __init__(self) -> None:
        json_file = open('/root/server/model/voice_analyzer/voice_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("/root/server/model/voice_analyzer/Emotion_Voice_Detection_Model.h5")
        print("Loaded model from disk")

    def evaluate(self, test_data):
        data, sampling_rate = librosa.load(test_data)
        X, sample_rate = librosa.load(test_data, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        featurelive = mfccs
        livedf2 = featurelive

        livedf2= pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim= np.expand_dims(livedf2, axis=2)
        livepreds = self.model.predict(twodim, 
                                batch_size=32, 
                                verbose=1)
        prediction=livepreds.argmax(axis=1)
        print(livepreds)
        print(prediction[0])

        return str(prediction[0])
        