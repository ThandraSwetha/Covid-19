from fastapi import FastAPI # it is used in ml , it is uesed to pridict your products clearly 
from pydantic import BaseModel # it enures data is clear and valid 
import joblib # to save and load your ml model
import numpy as np
import pandas as pd # to read/ upload our dataset
from typing import Literal 
from sklearn.preprocessing import LabelEncoder


ml = joblib.load("ml/covid_diag.pkl")

class inp_data(BaseModel):
    Age : int
    Gender : int
    Fever: int
    Cough: int
    Fatigue: int 
    Breathlessness: int
    Comorbidity : int
    Stage: int 
    Type : int 
    Tumor_Size : float

app = FastAPI()

@app.post('/')

def root_msg():
    return {'Hello' , 'Welcome'}

@app.post('/predict')

def prediction(data : inp_data):
    inputs = pd.DataFrame([data.dict()])
    
    prd = ml.predict(inputs)[0]

    # covert it into text 
    return {"Prediction ": prd}


