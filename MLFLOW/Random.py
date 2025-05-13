import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import os

# 1. تحميل البيانات
data_path = r"C:\Users\arwah\OneDrive\Desktop\HealthCare Project\datasets\cleaned_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found at: {data_path}")
data = pd.read_csv(data_path)
print("Columns in dataset:", data.columns)

# 2. تحديد الـ features والـ target
X = data.drop(columns=["Depression"])
y = data["Depression"]
X = X.fillna(0)

# 3. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. تطبيق SMOTE على بيانات التدريب
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 5. تحديد التجربة
mlflow.set_experiment("random_forest_exp_smote")

# 6. بدء الـ Run
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)

    signature = mlflow.models.signature.infer_signature(X_train, y_pred)
    input_example = X_train.iloc[0].to_dict()
    mlflow.sklearn.log_model(model, "random_forest_model", signature=signature, input_example=input_example)

    mlflow.set_tag("description", "Random Forest with SMOTE for Depression prediction")
    print(f"Logged model with ID: {mlflow.active_run().info.run_id}")