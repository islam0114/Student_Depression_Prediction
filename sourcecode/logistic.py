import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. تحميل البيانات من الباث المحدد
data_path = r"C:\Users\arwah\OneDrive\Desktop\HealthCare Project\datasets\cleaned_data.csv"  # ضع هنا الباث الصحيح للبيانات
data = pd.read_csv(data_path)

# 2. تحديد الأعمدة الخاصة بالـ features (X) والـ target (y)
X = data.drop(columns=["Depression"])  # استبدل "target_column" باسم عمود الـ target
y = data["Depression"]  # عمود الـ target

# 3. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. تحديد الـ Experiment في MLflow
mlflow.set_experiment("logisticcc")  # استبدل "your_experiment_name" باسم التجربة

# 5. بداية الـ run مع mlflow
with mlflow.start_run():

    # 6. تدريب الموديل
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 7. تسجيل الموديل في MLflow
    mlflow.sklearn.log_model(model, "logistic_regression_model")

    # 8. حساب accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 9. تسجيل الـ metrics (مثال: accuracy)
    mlflow.log_metric("accuracy", accuracy)

    # 10. تسجيل المعاملات (parameters) في الـ Run
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 200)

    # 11. تعريف signature و input_example (مهم علشان تتجنب التحذيرات في الـ UI)
    signature = mlflow.models.signature.infer_signature(X_train, y_train)
    input_example = X_train.iloc[0].to_dict()  # مثال من البيانات المدخلة
    mlflow.sklearn.log_model(model, "logistic_regression_model", signature=signature, input_example=input_example)

    print(f"Logged model with ID: {mlflow.active_run().info.run_id}")
