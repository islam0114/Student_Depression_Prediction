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

# تحميل البيانات
data_path = r"C:\Users\arwah\OneDrive\Desktop\HealthCare Project\datasets\cleaned_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found at: {data_path}")
df = pd.read_csv(data_path)

# تجهيز البيانات
X = df.drop(['Depression'], axis=1).fillna(0)
y = df['Depression']

# تطبيق SMOTEENN على كل البيانات (كما في Jupyter)
smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
X_smote, y_smote = smote_enn.fit_resample(X, y)

# تقسيم البيانات بعد التوازن
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=1)

# تحديد تجربة MLflow
mlflow.set_experiment("xgboost_exp_with_matrix_plot")

with mlflow.start_run():
    # تدريب النموذج
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)

    # التنبؤ
    y_pred = xgb.predict(X_test)
    train_acc = accuracy_score(y_train, xgb.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    # تسجيل المقاييس
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    # classification report
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    with open("confusion_matrix.txt", "w") as f:
        f.write(str(cm))
    mlflow.log_artifact("confusion_matrix.txt")

    # ⬇️ رسم confusion matrix كصورة
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix_plot.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix_plot.png")

    # تسجيل النموذج
    signature = mlflow.models.signature.infer_signature(X_test, y_pred)
    input_example = X_test.iloc[0].to_dict()
    mlflow.xgboost.log_model(xgb, "xgboost_model", signature=signature, input_example=input_example)

    mlflow.set_tag("description", "XGBoost with SMOTEENN on full data and matrix plot")
    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
