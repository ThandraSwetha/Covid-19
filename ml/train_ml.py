import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
import joblib

df = pd.read_csv("D:\\deno-covid\\Data\\pneumonia_covid_diagnosis_dataset.csv")
print(df.head())


cols = ["Gender", "Fever", "Cough", "Fatigue", "Breathlessness","Comorbidity","Stage", "Type"]

for i in cols:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])


df = df.drop("Is_Curable", axis = 1)


# df= pd.get_dummies(columns=["Stage"], drop_first = True)
# it is another type to replace the labelEncoder 
# all the models will be present in esemble 
#Randomforest i sboth classification , and regressiion 

x = df.drop(columns=["Survival_Rate"], axis = 1)
y = df['Survival_Rate']



x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2 , random_state=42)

model = RandomForestRegressor()
model.fit(x_train,y_train)

# in enpochs = nueral networks 

# to save the model 

prd = model.predict(x_test)

joblib.dump(model, "covid_diag.pkl")




