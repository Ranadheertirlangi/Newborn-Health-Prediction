# Newborn-Health-Prediction
This project aims to develop an intelligent system that predicts the health condition of newborn babies based on medical and physiological parameters. The goal is to assist healthcare professionals in identifying potential health risks at an early stage and improve neonatal care outcomes.

### Objective
To analyze newborn health data and identify factors influencing health outcomes.

To build a predictive machine learning model that classifies the health status of a newborn as Healthy or At Risk.

To provide an interpretable model that can support early diagnosis and intervention.

### Technologies Used

Python

Pandas, NumPy – Data processing

Matplotlib, Seaborn – Data visualization

Scikit-learn – Machine learning modeling

Imbalanced-learn (imblearn) – Oversampling using SMOTE

XGBoost, Random Forest, Logistic Regression – Classification models

## Model Development Process

### 1. Data Preprocessing

Managed missing values and handled outliers.

Encoded categorical variables and normalized numerical features.

Balanced the dataset using SMOTE to address class imbalance.

### 2. Exploratory Data Analysis (EDA)

Visualized feature distributions and correlations.

Identified key indicators affecting newborn health outcomes.

### 3. Model Training & Evaluation

Implemented and compared multiple models:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Evaluated performance using metrics like Accuracy, Precision, Recall, and F1-Score.

Visualized model performance with Confusion Matrix and ROC Curve.
