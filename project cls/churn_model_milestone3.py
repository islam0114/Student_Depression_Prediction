
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Set random seed
RANDOM_STATE = 42

# Load datasets
cleaned_df = pd.read_csv(r"C:\Users\Maryam\Downloads\cleaned_data.csv")
standardized_df = pd.read_csv(r"C:\Users\Maryam\Downloads\standardized_data.csv")
normalized_df = pd.read_csv(r"C:\Users\Maryam\Downloads\normalized_data.csv")

datasets = {
    "Cleaned": cleaned_df,
    "Standardized": standardized_df,
    "Normalized": normalized_df
}

# Parameter grids
param_grids = {
    "Decision Tree": {
        "max_depth": [5, 10, 15],
        "criterion": ["gini", "entropy"]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    }
}

# Evaluate function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# Results container
all_results = []

# Process each dataset
for name, df in datasets.items():
    print(f"\nProcessing dataset: {name}")
    X = df.drop("Depression", axis=1)
    y = df["Depression"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Apply SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Model definitions
    models = {
        "Decision Tree": GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), param_grids["Decision Tree"], cv=5),
        "Naive Bayes": GaussianNB(),
        "KNN": GridSearchCV(KNeighborsClassifier(), param_grids["KNN"], cv=5)
    }

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name} on {name}")
        model.fit(X_train_resampled, y_train_resampled)
        result = evaluate_model(model, X_test, y_test)
        best_params = model.best_params_ if hasattr(model, "best_params_") else "Default"
        result.update({
            "Dataset": name,
            "Model": model_name,
            "Best Params": best_params
        })
        all_results.append(result)

# Output results
results_df = pd.DataFrame(all_results)
results_df.to_csv("churn_model_evaluation_results.csv", index=False)
print("\nAll model results saved to churn_model_evaluation_results.csv")
