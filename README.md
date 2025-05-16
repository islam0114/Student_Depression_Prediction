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


2.1 Data Collection & Analysis
Gather public or survey-based datasets related to student mental health
Clean and preprocess the data to ensure consistency
Explore the dataset through Exploratory Data Analysis (EDA)

2.2 Identifying Risk Factors
Analyze connections between academic, social, and personal variables
Use statistical analysis and feature selection to highlight significant predictors

2.3 Machine Learning for Prediction
Build predictive models to classify students into:
1)Depressed 
2)not Depressed

Compare the performance of models like:
Logistic Regression
Decision Trees
Random Forest
lightGBM
XGB
Neural Networks
SVM
GradientBoosting
KNN

2.4 Data Visualization & Reporting
Use tools like Matplotlib, Seaborn,and Power BI
Create interactive dashboards and clear visual summaries
Provide practical recommendations based on data insights

3.  Scope
 Included
Public dataset collection and preparation
Exploratory data analysis (EDA)
Machine learning model development
Visual presentation of findings

 Excluded
Real-time diagnosis or clinical recommendations
Clinical assessment data (unless publicly available)
Psychological counseling or direct mental health services

4.  Technologies Used
Category	Tools / Libraries
Programming Language	Python
Data Analysis	pandas, NumPy
Machine Learning	scikit-learn, 
Visualization	Matplotlib, Seaborn,Power BI

5.  Expected Outcomes
Identification of key factors linked to student depression
Clear classification of students into risk levels
Dashboards and visuals to help universities make informed decisions
Data-backed recommendations to support mental health programs

6. Outcome
This project provides a data-driven understanding of depression among students. The insights gained can assist academic institutions in identifying at-risk individuals and designing preventive mental health programs. While it is not a substitute for medical advice, the analysis offers a valuable foundation for further research and intervention planning.
