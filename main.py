import json, joblib, warnings, os, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MedGuard AI API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading models...")
diabetes_model        = tf.keras.models.load_model("diabetes_model_v2.keras")
diabetes_preprocessor = joblib.load("diabetes_preprocessor_v2.pkl")
with open("diabetes_threshold_v2.json") as f:
    D_THRESHOLD = json.load(f)["threshold"]

heart_pipeline = joblib.load("heart_model_v2.pkl")
with open("heart_threshold_v2.json") as f:
    H_THRESHOLD = json.load(f)["threshold"]

xray_model     = tf.keras.models.load_model("models\\xray_model.keras")
XRAY_THRESHOLD = 0.70
XRAY_IMG_SIZE  = (224, 224)

heart_rf        = heart_pipeline.named_steps["model"]
heart_explainer = shap.TreeExplainer(heart_rf)

print(f"Diabetes  loaded | Threshold: {D_THRESHOLD}")
print(f"Heart     loaded | Threshold: {H_THRESHOLD}")
print(f"X-Ray     loaded!")
print("All models ready!")

class DiabetesInput(BaseModel):
    Pregnancies: float; Glucose: float; BloodPressure: float
    SkinThickness: float; Insulin: float; BMI: float
    DiabetesPedigreeFunction: float; Age: float

class HeartInput(BaseModel):
    HighBP: float; HighChol: float; CholCheck: float; BMI: float
    Smoker: float; Stroke: float; Diabetes: float; PhysActivity: float
    Fruits: float; Veggies: float; HvyAlcoholConsump: float
    AnyHealthcare: float; NoDocbcCost: float; GenHlth: float
    MentHlth: float; PhysHlth: float; DiffWalk: float; Sex: float
    Age: float; Education: float; Income: float

class CombinedInput(BaseModel):
    diabetes: DiabetesInput
    heart: HeartInput

def get_risk_zone(score):
    if score >= 65: return "RED"
    if score >= 35: return "YELLOW"
    return "GREEN"

def prep_diabetes(data):
    d = data.dict()
    df = pd.DataFrame([d])
    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    df["BMI_Age"]         = df["BMI"] / (df["Age"] + 1)
    df["Glucose_Insulin"] = df["Glucose"] * df["Insulin"]
    df["BP_Age"]          = df["BloodPressure"] / (df["Age"] + 1)
    return diabetes_preprocessor.transform(df)

def prep_heart(data):
    return pd.DataFrame([data.dict()])

def get_shap_diabetes(X_prep, feature_names):
    bg = np.zeros((1, X_prep.shape[1]))
    explainer = shap.KernelExplainer(lambda x: diabetes_model.predict(x, verbose=0).ravel(), bg)
    shap_vals = explainer.shap_values(X_prep, nsamples=50)
    shap_arr  = np.array(shap_vals).flatten()[:len(feature_names)]
    top = sorted(zip(feature_names, shap_arr), key=lambda x: abs(x[1]), reverse=True)[:6]
    return [{"feature": f, "shap_value": round(float(s),4), "impact": "increases_risk" if s>0 else "decreases_risk"} for f,s in top]

def get_shap_heart(X_df, feature_names):
    hp     = heart_pipeline.named_steps.get("preprocess", heart_pipeline.named_steps.get("scaler"))
    X_prep = hp.transform(X_df)
    sv     = heart_explainer.shap_values(X_prep)
    if isinstance(sv, np.ndarray) and sv.ndim==3: shap_arr = sv[0,:,1]
    elif isinstance(sv, list): shap_arr = np.array(sv[1]).flatten()
    else: shap_arr = np.array(sv).flatten()
    top = sorted(zip(feature_names, shap_arr), key=lambda x: abs(x[1]), reverse=True)[:6]
    return [{"feature": f, "shap_value": round(float(s),4), "impact": "increases_risk" if s>0 else "decreases_risk"} for f,s in top]

def prep_xray(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(XRAY_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

@app.get("/")
def root(): return {"message": "MedGuard AI API v2.0"}

@app.get("/health")
def health():
    return {"status":"ok","models":{"diabetes":"loaded","heart":"loaded","xray":"loaded"},"thresholds":{"diabetes":D_THRESHOLD,"heart":H_THRESHOLD,"xray":XRAY_THRESHOLD}}

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        feature_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","BMI_Age","Glucose_Insulin","BP_Age"]
        X_prep = prep_diabetes(data)
        prob   = float(diabetes_model.predict(X_prep, verbose=0).ravel()[0])
        score  = round(prob*100,1)
        return {"model":"diabetes","probability":score,"prediction":"Diabetic" if prob>=D_THRESHOLD else "Not Diabetic","risk_zone":get_risk_zone(score),"threshold":D_THRESHOLD,"top_factors":get_shap_diabetes(X_prep,feature_names)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    try:
        feature_names = list(data.dict().keys())
        X_df  = prep_heart(data)
        prob  = float(heart_pipeline.predict_proba(X_df)[0][1])
        score = round(prob*100,1)
        return {"model":"heart_disease","probability":score,"prediction":"Heart Disease" if prob>=H_THRESHOLD else "No Disease","risk_zone":get_risk_zone(score),"threshold":H_THRESHOLD,"top_factors":get_shap_heart(X_df,feature_names)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/xray")
async def predict_xray(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        X_img = prep_xray(image_bytes)
        prob  = float(xray_model.predict(X_img, verbose=0).ravel()[0])
        pred  = int(prob >= XRAY_THRESHOLD)
        score = round(prob*100,1)
        prediction = "PNEUMONIA" if pred==1 else "NORMAL"
        return {"model":"xray_pneumonia","probability":score,"prediction":prediction,"risk_zone":"RED" if pred==1 else "GREEN","confidence":score if pred==1 else round((1-prob)*100,1),"threshold":XRAY_THRESHOLD}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/combined")
def predict_combined(data: CombinedInput):
    try:
        d_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","BMI_Age","Glucose_Insulin","BP_Age"]
        X_d    = prep_diabetes(data.diabetes)
        d_prob = float(diabetes_model.predict(X_d, verbose=0).ravel()[0])
        d_pred = int(d_prob >= D_THRESHOLD)
        X_h    = prep_heart(data.heart)
        h_prob = float(heart_pipeline.predict_proba(X_h)[0][1])
        h_pred = int(h_prob >= H_THRESHOLD)
        combined = round((d_prob*0.5 + h_prob*0.5)*100,1)
        zone = get_risk_zone(combined)
        return {"risk_score":combined,"risk_zone":zone,"alert":zone=="RED",
            "diabetes":{"probability":round(d_prob*100,1),"prediction":"Diabetic" if d_pred else "Not Diabetic","risk_zone":get_risk_zone(round(d_prob*100,1)),"top_factors":get_shap_diabetes(X_d,d_names)},
            "heart_disease":{"probability":round(h_prob*100,1),"prediction":"Heart Disease" if h_pred else "No Disease","risk_zone":get_risk_zone(round(h_prob*100,1)),"top_factors":get_shap_heart(X_h,list(data.heart.dict().keys()))}}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
