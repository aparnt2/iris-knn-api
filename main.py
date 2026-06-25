from fastapi import FastAPI 
from pydantic import BaseModel
from sklearn.datasets import load_iris
import joblib

app=FastAPI()
irisdata=load_iris()
model=joblib.load('models/iris_model.joblib')
scaler=joblib.load("models/iris_scaler.joblib")

class Iris(BaseModel):
    sepal_length  :int
    sepal_width:int
    petal_length:int
    petal_width:int

@app.post('/predict')
def prediction(iris :Iris):
    sample=[[iris.sepal_length,iris.sepal_width,iris.petal_length,iris.petal_width]]
    sample=scaler.transform(sample)
    prediction=model.predict(sample)
    result=irisdata.target_names[prediction[0]]
    return{
        "prediction":result
    }

