from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load("house_model.pkl")


class House(BaseModel):
    area: int
    bedrooms: int


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(data: House):

    prediction = model.predict(
        [[data.area, data.bedrooms]]
    )

    return {
        "predicted_price": float(prediction[0])
    }