from pydantic import BaseModel

class CropData(BaseModel):
    N: int               # Cantidad de Nitrógeno en el suelo
    P: int               # Cantidad de Fósforo en el suelo
    K: int               # Cantidad de Potasio en el suelo
    temperature: float   # Temperatura (°C)
    humidity: float      # Humedad relativa (%)
    ph: float            # pH del suelo
    rainfall: float      # Lluvia (mm)
