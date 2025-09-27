import numpy as np
import pickle
from schemas.crop_schemas import CropData 

# Cargamos el modelo de RandomForest entrenado para cultivos
with open('RFCropModel.pkl', 'rb') as file:   # asegúrate de guardar tu modelo con este nombre
    RF_model = pickle.load(file)

def crop_prediction(data: CropData):
    # Preparamos los datos de entrada en el orden correcto
    xin = np.array([
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]).reshape(1, -1)   # 7 features

    # Predicción
    prediction = RF_model.predict(xin)
    return prediction[0]
