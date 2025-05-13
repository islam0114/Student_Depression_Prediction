import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 1. تحميل البيانات من الباث المحدد
data_path = r"C:\Users\arwah\OneDrive\Desktop\HealthCare Project\datasets\cleaned_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found at: {data_path}")

data = pd.read_csv(data_path)

# 2. التحقق من أسماء الأعمدة
print("Columns in dataset:", data.columns)

# 3. تحديد الأعمدة الخاصة بالـ features (X) والـ target (y)
X = data.drop(columns=["Depression"])
y = data["Depression"]

# 4. معالجة القيم المفقودة
X = X.fillna(0)  # أو استخدم طرق أخرى حسب الحاجة

# 5. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. تحديد الـ Experiment في MLflow
mlflow.set_experiment("lightgbm_exp")

# 7. بداية الـ run مع mlflow
with mlflow.start_run():
    # 8. تدريب الموديل
    model = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # 9. حساب المقاييس
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 10. تسجيل الـ metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # 11. تسجيل المعاملات
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("random_state", 42)

    # 12. تعريف signature و input_example
    signature = mlflow.models.signature.infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[0].to_dict()

    # 13. تسجيل الموديل
    mlflow.lightgbm.log_model(model, "lightgbm_model", signature=signature, input_example=input_example)

    # 14. إضافة وصف للـ Run
    mlflow.set_tag("description", "LightGBM model for Depression prediction")

    print(f"Logged model with ID: {mlflow.active_run().info.run_id}")