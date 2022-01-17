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

        result = prediction[0]
        print(result)

        #label-list
        # 0 - female_angry /
        # 1 - female_calm
        # 2 - female_fearful
        # 3 - female_happy /
        # 4 - female_sad
        # 5 - male_angry /
        # 6 - male_calm
        # 7 - male_fearful
        # 8 - male_happy /
        # 9 - male_sad

        final_result = str(-1)

        #angry
        if(result==0 or result==5):
            final_result = str(0)
        #calm
        elif(result==1 or result==6):
            final_result = str(1)
        #fearful
        elif(result==2 or result==7):
            final_result = str(2)
        #happy
        elif(result==3 or result==8):
            final_result = str(3)
        #sad
        elif(result==4 or result==9):
            final_result = str(4)


        # #angry, happy
        # if(result==0 or result==3 or result==5 or result==8):
        #     final_result = str(1)
        # #calm, fearful, sad
        # else:
        #     final_result = str(0)

        return final_result
        