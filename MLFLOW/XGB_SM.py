import mlflow
import mlflow.xgboost
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score


data_path = r"C:\Users\arwah\OneDrive\Desktop\HealthCare Project\datasets\cleaned_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found at: {data_path}")
df = pd.read_csv(data_path)


X = df.drop(['Depression'], axis=1).fillna(0)
y = df['Depression']


smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
X_smote, y_smote = smote_enn.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=1)


mlflow.set_experiment("xgb3")

with mlflow.start_run():
    
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)

  
    y_pred = xgb.predict(X_test)
    train_acc = accuracy_score(y_train, xgb.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)


    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

 
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
