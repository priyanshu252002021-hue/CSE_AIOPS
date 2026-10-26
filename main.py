
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load trained model from models folder
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "iris_model.pkl"

model = joblib.load(MODEL_PATH)

# Create FastAPI application
app = FastAPI(title="Iris Random Forest API")


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Iris API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    species = ["setosa", "versicolor", "virginica"]

    return {
        "prediction": prediction,
        "class_name": species[prediction]
    }
