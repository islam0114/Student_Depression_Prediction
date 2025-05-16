import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.combine import SMOTEENN
import os


data_path = r"C:\Users\arwah\OneDrive\Desktop\HealthCare Project\datasets\cleaned_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found at: {data_path}")
df = pd.read_csv(data_path)
print("Columns in dataset:", df.columns)


X = df.drop(['Depression'], axis=1)
y = df['Depression']



X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Original Test samples:", len(X_test))
print(y_test.value_counts())


smote_enn = SMOTEENN(random_state=42, n_jobs=-1)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_orig, y_train_orig)
print(f'Resampled dataset shape: {pd.Series(y_train_resampled).value_counts(normalize=True)}')


mlflow.set_experiment("xgboost_exp_smoteenn")


with mlflow.start_run():
    
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train_resampled, y_train_resampled)

    
    train_accuracy = accuracy_score(y_train_resampled, xgb.predict(X_train_resampled))
    test_accuracy = accuracy_score(y_test, xgb.predict(X_test))
    y_pred_xgb = xgb.predict(X_test)

    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("xgboost_accuracy", accuracy_score(y_test, y_pred_xgb))

    
    report = classification_report(y_test, y_pred_xgb)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    
    cm = confusion_matrix(y_test, y_pred_xgb)
    with open("confusion_matrix.txt", "w") as f:
        f.write(str(cm))
    mlflow.log_artifact("confusion_matrix.txt")

    
    signature = mlflow.models.signature.infer_signature(X_test, y_pred_xgb)
    input_example = X_test.iloc[0].to_dict()
    mlflow.xgboost.log_model(xgb, "xgboost_model", signature=signature, input_example=input_example)

    mlflow.set_tag("description", "XGBoost with SMOTEENN for Depression prediction")

    print(f"Logged model with ID: {mlflow.active_run().info.run_id}")
    print("XGB Training Accuracy:", train_accuracy)
    print("XGB Test Accuracy:", test_accuracy)
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
