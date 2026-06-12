from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(
    title="Hospital Readmission API",
    version="1.0"
)

print("Loading model...")

model = joblib.load(
    "models/xgboost_model.pkl"
)

print("Model Loaded")

@app.get("/")
def health():

    return {
        "status": "running"
    }


@app.post("/predict")
def predict(payload: dict):

    df = pd.DataFrame(
        [payload]
    )

    probability = (
        model.predict_proba(df)
    )[0][1]

    prediction = int(
        probability >= 0.5
    )

    return {
        "risk_score": round(
            float(probability),
            4
        ),
        "prediction": prediction
    }