from fastapi import APIRouter
from schemas.crop_schemas import CropData
from services.crop_service import crop_prediction

router = APIRouter()

@router.post("/predict")
async def cropPredict(data: CropData, model: str = "rf"):
    print("model used ", model)
    prediction = crop_prediction(data, model_name=model)

    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()


    if isinstance(prediction, (list, tuple)):
        if len(prediction) == 1:
            prediction = prediction[0]

    prediction = str(prediction)
    return {
        "model_used": model,
        "prediction": prediction
    }
