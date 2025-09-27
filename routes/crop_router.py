from fastapi import APIRouter
from schemas.crop_schemas import CropData
from services.crop_service import crop_prediction

router = APIRouter()

@router.post("/predict")
async def cropPredict(data: CropData):
    prediction = crop_prediction(data)
    return {"prediction": prediction}
