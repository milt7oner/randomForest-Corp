import numpy as np
import pickle
from schemas.crop_schemas import CropData 


with open('RFCropModel.pkl', 'rb') as file:    
    RF_model = pickle.load(file)

def crop_prediction(data: CropData):
 
    xin = np.array([
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]).reshape(1, -1)  

   
    prediction = RF_model.predict(xin)
    return prediction[0]
