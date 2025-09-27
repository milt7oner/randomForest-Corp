import numpy as np
import pickle
from schemas.crop_schemas import CropData

# Cargamos los modelos entrenados
with open('RFCropModel.pkl', 'rb') as file:    
    RF_model = pickle.load(file)

with open('SVCModel.pkl', 'rb') as file:
    SVC_model = pickle.load(file)

def crop_prediction(data: CropData, model_name: str = "rf"):
    """
    Hace la predicción usando el modelo especificado.
    model_name puede ser 'rf' (RandomForest) o 'svc' (SVM).
    """
    xin = np.array([
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]).reshape(1, -1)  

    if model_name == "rf":
        prediction = RF_model.predict(xin)
    elif model_name == "svc":
        prediction = SVC_model.predict(xin)
    else:
        raise ValueError("Modelo no soportado. Usa 'rf' o 'svc'.")

    return prediction
