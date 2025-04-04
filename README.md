# Diabetes-Prediction
# 🩺 Diabetes Prediction Using Machine Learning

## Overview
This project focuses on predicting the likelihood of diabetes in individuals using machine learning algorithms. It leverages the **Pima Indians Diabetes Dataset**, which contains various medical attributes relevant to diabetes diagnosis.

## 🔍 Problem Statement
Early diagnosis of diabetes can prevent severe health issues. The goal is to develop a predictive model that can assist healthcare professionals by accurately classifying patients as diabetic or non-diabetic based on medical features.

## 📊 Dataset
- **Source:** [Kaggle - Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Attributes:**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
  - Outcome (0 = Non-Diabetic, 1 = Diabetic)

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## 📌 Project Workflow

1. **Data Preprocessing**
   - Handling missing/zero values
   - Feature scaling (StandardScaler)
   - Train-test split (80-20)

2. **Exploratory Data Analysis (EDA)**
   - Correlation heatmap
   - Distribution plots

3. **Model Building**
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)

4. **Evaluation Metrics**
   - Accuracy Score
   - Confusion Matrix
   - ROC-AUC Curve
   - Precision, Recall, F1-Score

## 🧠 Best Model Performance
| Model                | Accuracy |
|---------------------|----------|
| Random Forest        | 81.8%    |
| Logistic Regression  | 78.4%    |
| SVM (RBF Kernel)     | 79.2%    |

> *Random Forest achieved the best performance with balanced precision and recall.*

## 📁 Folder Structure
├── data/ │ └── diabetes.csv ├── notebooks/ │ └── Diabetes_Prediction.ipynb ├── models/ │ └── random_forest.pkl ├── README.md └── requirements.txt

## 🚀 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
Install dependencies
pip install -r requirements.txt
Launch the notebook
jupyter notebook notebooks/Diabetes_Prediction.ipynb
📈 Future Improvements

Deploy using Flask or Streamlit
Hyperparameter tuning (GridSearchCV)
Model explainability (SHAP, LIME)
📬 Contact

Rahul Pandey
📧 rahulpandey02124@gmail.com
📍 India
