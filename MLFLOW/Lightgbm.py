import mlflow
import mlflow.lightgbm
import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models.signature import infer_signature

# تحميل البيانات
data_path = r"C:\Users\arwah\OneDrive\Desktop\HealthCare Project\datasets\cleaned_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found at: {data_path}")
df = pd.read_csv(data_path)

# تجهيز البيانات
X = df.drop(['Depression'], axis=1).fillna(0)
y = df['Depression']

# تطبيق SMOTEENN
smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
X_smote, y_smote = smote_enn.fit_resample(X, y)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=1)

# تحديد تجربة MLflow
mlflow.set_experiment("lightgbm_smote")

with mlflow.start_run():
    # تدريب النموذج
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # التنبؤ
    y_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    # تسجيل المقاييس
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # رسم Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("lgbm_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("lgbm_confusion_matrix.png")

    # تسجيل النموذج
    signature = infer_signature(X_test, y_pred)
    input_example = X_test.iloc[0].to_dict()
    mlflow.lightgbm.log_model(model, "lightgbm_model", signature=signature, input_example=input_example)

    # تسجيل المعاملات
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("learning_rate", 0.1)

    mlflow.set_tag("description", "LightGBM with SMOTEENN and metrics logged to MLflow")

    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
    print("Confusion Matrix:\n", cm)
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[0:1]

    # تسجيل موديل LightGBM
    mlflow.lightgbm.log_model(
        model,
        artifact_path="lgbm_model",
        signature=signature,
        input_example=input_example
    )
