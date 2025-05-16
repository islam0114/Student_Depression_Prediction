# Student Depression Analysis Using Data Science
This project aims to develop a predictive model to identify university students at risk of depression using psychological, academic, and lifestyle factors. The goal is to provide educational institutions with data-driven insights to support mental health strategies and early intervention efforts.

----

## 1. Project Overview
University students face increasing mental health challenges, with depression being one of the most widespread disorders. This project leverages data science techniques to:

- Identify critical risk factors for student depression

- Build machine learning models to classify students by risk level

- Offer actionable insights for improving campus mental health services

----

## 2. Team Members
- Arwa Hamdy Mohammdy

- Eslam Mohammed Abdelfattah

- Mariam Mostafa Abdelaziz

- Mariam Mostafa Abdelaal

----

## 3. Dataset Description
Source: Kaggle (after over a month of exploration)

The dataset includes a wide variety of features categorized into:

- Personal Information:
ID, gender, age, city, pet ownership, favorite color

- Academic Factors:
Academic pressure, CGPA, degree, study satisfaction, study/work hours

- Psychological Factors:
Suicidal thoughts, depression status, PHQ-9 score, cortisol level

- Social & Emotional Factors:
Financial stress, family history of mental illness, social isolation, bullying, family issues, uncertain future, social media usage

- Lifestyle & Health Habits:
Sleep duration, dietary habits, drug/smoking, daily coffee intake, music genre preference

----

## 4. Data Preprocessing

- Missing Values: Only 3 missing values (in Financial Stress) were dropped

- Categorical Encoding: Applied Label Encoding to convert categorical features

- Feature Scaling: Used for models sensitive to feature scale (e.g., SVM, Logistic Regression)

- Train-Test Split: 70% training / 30% testing

----

## 5. Model Development
Multiple machine learning models were tested. XGBoost was selected as the final model due to its:

- High accuracy

- Robust performance on imbalanced data

- Capability to capture both linear and non-linear patterns

- Built-in support for feature importance analysis

The final XGBoost model was saved and prepared for deployment.

----

## 6. Model Deployment
Input Features:

- Academic Pressure (0–5)

- Study Satisfaction (0–5)

- Dietary Habits (Healthy/Unhealthy)

- Suicidal Thoughts (Yes/No)

- Study Hours per Day (0–12)

- Financial Stress (0–5)

- Bullying (Yes/No)

- PHQ-9 Score (0–20)

- Cortisol Level (0–10)

----

## 7. Challenges Faced
- Data Collection: Time-consuming and required extensive search

- Missing Values: Minimal, handled by row deletion

- Outliers: Retained PHQ-9 outliers (<5%) for data distribution integrity

- Imbalanced Target: Required metrics like F1-score and ROC-AUC

- Feature Selection for Plots: Needed to avoid noisy or uninformative visuals

- High Dimensionality: Increased model complexity and training time

- Scaling Issues: Addressed normalization for better visualization and modeling

- Text Data: Required appropriate encoding without losing semantics

- Overfitting Risk: Mitigated with model tuning and feature control

-----

## 8. Key Insights
- Critical Risk Factors Identified: e.g., age, PHQ-9 score, mental health history

- Explainable Models: Feature importance helped interpret predictions

- Proactive Decision Support: Early identification enables better mental health planning

- Deployment-Ready Architecture: Model is scalable and API/web compatible

- Balanced Evaluation Metrics: Focused on precision, recall, F1-score, ROC-AUC

- Insightful Visualizations: Helped uncover patterns and trends in student behavior

----

## 9. Integration Recommendations
- EHR Integration: Embed risk predictions into student health systems

- Clinical Decision Support: Show risk scores + suggested actions

- Workflow Automation: Trigger alerts or follow-up scheduling for high-risk students

- Feedback Loop: Enable updates and corrections to the model

- Staff Training: Educate staff on model use and interpretation

- Compliance: Ensure fairness, bias audits, and transparency in usage

- Collaborative Use: Promote use in multidisciplinary health or counseling teams

----

## 10. Tools & Technologies

| Category         | Tools                         |
| ---------------- | ----------------------------- |
| Language         | Python                        |
| Data Processing  | pandas, NumPy                 |
| Machine Learning | scikit-learn, XGBoost         |
| Visualization    | Matplotlib, Seaborn, Power BI |

----

## 11. Project Structure
Health-Care-Project/

│

├── APP/                           # App-related files (possibly UI or interface)

│

├── CODE/                          # Final scripts and source code

│

├── DATASETS/                      # Processed datasets

│

├── DEPLOYMENT/                    # Deployment logic or configuration

│

├── Dashboard/                     # Power BI dashboards and reports

│

├── MLFLOW/                        # MLflow-related tracking folders

│

├── PRESENTATION/                 # Presentation slides

│

├── REPORTS/                       # Final reports and documentation

│

├── catboost_info/                # Logs or model info from CatBoost (auto-generated)

│

├── mlruns/                        # MLflow tracking directory (auto-generated)

│

├── README.md                      # Project documentation and overview

├── requirements.txt               # List of required Python packages

