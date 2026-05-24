from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("house_model.pkl")

class House(BaseModel):
    area:int
    bedrooms:int


@app.post("/predict")
def predict(data: House):

    prediction = model.predict(
        [[data.area, data.bedrooms]]
    )

    return {
        "predicted_price": float(prediction[0])
    }
@app.get("/")
def home():
    return {
        "message":"House Price Prediction API is running"
    }