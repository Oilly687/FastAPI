from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np
from sklearn.datasets import load_iris
from datetime import datetime

# โหลดโมเดลที่เทรนแล้ว
model = joblib.load("iris_model.pkl")

app = FastAPI()

# เชื่อมต่อ Firebase
cred = credentials.Certificate("emotibit01-firebase-adminsdk-y12ef-e59ba922b4.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://emotibit01-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

class IrisPrediction(BaseModel):
    predicted_class: int
    predicted_class_name: str
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    prediction_time: str

@app.get("/predict-from-firebase", response_model=IrisPrediction)
async def predict_from_firebase():
    try:
        ref = db.reference("/iris_input/raw")
        input_data = ref.get()

        if not input_data:
            return JSONResponse(content={"error": "No data found in Firebase"}, status_code=404)

        latest_timestamp = max(input_data.keys())

        input_data_values = input_data[latest_timestamp]

        sepal_length = float(input_data_values["sepal_length"])
        sepal_width = float(input_data_values["sepal_width"])
        petal_length = float(input_data_values["petal_length"])
        petal_width = float(input_data_values["petal_width"])

        input_data_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        predicted_class = model.predict(input_data_array)[0]
        predicted_class_name = load_iris().target_names[predicted_class]

        prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "predicted_class": int(predicted_class),
            "predicted_class_name": str(predicted_class_name),
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
            "prediction_time": prediction_time,
        }

        predict_ref = db.reference("/iris_input/predict")
        predict_ref.push(result)

        return JSONResponse(content=result)

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
