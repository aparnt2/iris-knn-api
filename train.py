import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import joblib

iris=load_iris()

df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df["species"]=iris.target

X=df.drop(columns="species")
y=df['species']
scaler=StandardScaler()

print(iris.target_names)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train_scalar=scaler.fit_transform(X_train)
X_test_scalar=scaler.transform(X_test)

model=KNeighborsClassifier(n_neighbors=3)

model.fit(X_train_scalar,y_train)
prediction=model.predict(X_test_scalar)

print(f"Accuracy:\n{accuracy_score(y_test, prediction):.2f}")

print("confusion matrix",confusion_matrix(y_test,prediction))

joblib.dump(model,'models/iris_model.joblib')
joblib.dump(scaler,'models/iris_scaler.joblib')







