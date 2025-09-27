from fastapi import APIRouter
from schemas.crop_schemas import CropData
from services.crop_service import crop_prediction

router = APIRouter()

@router.post("/predict")
async def cropPredict(data: CropData, model: str = "rf"):
    # Llamamos a la función de predicción
    prediction = crop_prediction(data, model_name=model)

    # Si la predicción viene como numpy array → convertir
    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()

    # Si es lista/array → tomar primer valor
    if isinstance(prediction, (list, tuple)):
        if len(prediction) == 1:
            prediction = prediction[0]

    # Asegurar que sea string (ejemplo: "maíz", "rice")
    prediction = str(prediction)

    return {
        "model_used": model,
        "prediction": prediction
    }
