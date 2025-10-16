# Random_Forest

# 🏦 Loan Approval Prediction using Machine Learning
## 📘 Project Overview

This project predicts loan approval based on an applicant’s financial and demographic details 

using multiple machine learning algorithms.

It includes complete data preprocessing, outlier detection, normalization, feature encoding, 

and model evaluation with visualizations.

## Loan Prediction App:



## 📊 Dataset

### Dataset used: loan_data.csv

Each row represents a loan application record.

### Feature	Description
person_age	Age of applicant
person_income	Annual income
person_emp_exp	Employment experience (years)
person_home_ownership	Home ownership status
loan_intent	Purpose of the loan
loan_amnt	Requested loan amount
loan_int_rate	Interest rate of loan
loan_percent_income	Loan amount as % of income
cb_person_cred_hist_length	Credit history length
credit_score	Applicant’s credit score
previous_loan_defaults_on_file	Whether previous loan was defaulted
loan_status	Target (1 = Approved, 0 = Rejected)
## ⚙️ Steps Performed

### 1️⃣ Data Preprocessing

Checked for missing values

Filled missing numeric values with median and categorical values with mode

Converted categorical data into numeric using Label Encoding

### 2️⃣ Exploratory Data Analysis (EDA)

Summary statistics (info(), describe())

Correlation heatmap

Histograms and boxplots for distribution visualization

Outlier detection using IQR Method

### 3️⃣ Feature Transformation

Applied PowerTransformer (Yeo–Johnson) to handle skewness

Visualized before and after transformation histograms and boxplots

### 4️⃣ Outlier Handling

Detected outliers using IQR

Capped extreme values within acceptable range

### 5️⃣ Model Training

Split dataset:

80% → Training  
20% → Testing

 Trained models:

Model	Description
🎯 Random Forest	Robust ensemble classifier
🧠 Logistic Regression	Interpretable baseline model
🌳 Decision Tree	Easy to visualize and explain
🔷 SVM	Linear decision boundary visualization

### 6️⃣ Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Actual vs Predicted scatter plots

R² Score (for Logistic Regression)

## 📈 Visualizations

### Histograms & boxplots (before/after normalization)

#### Before Normalization
<img width="1187" height="3190" alt="image" src="https://github.com/user-attachments/assets/619fc331-16d0-40a6-a8a3-547c5dddedbc" />

#### After Normalization
<img width="1189" height="3190" alt="image" src="https://github.com/user-attachments/assets/7bc98a69-58be-48a2-8b98-63a1aa95567f" />

### Confusion matrix heatmaps
<img width="528" height="435" alt="image" src="https://github.com/user-attachments/assets/f14431f8-9235-43a2-8d4e-43c8856e7bda" />

### Decision tree plot
<img width="1570" height="812" alt="image" src="https://github.com/user-attachments/assets/9a8340dc-d33a-4401-9511-8c48093e2b5d" />

### SVM decision boundary
<img width="565" height="455" alt="image" src="https://github.com/user-attachments/assets/a18e31aa-d0d5-40ca-96f7-783f3bf5193d" />

### Scatter plot (Actual vs Predicted)
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/866b9937-bbdc-4747-b0ad-5bde593eb76b" />

## 🧩 Technologies Used
Category	Tools
Programming Language	Python
Libraries	pandas, numpy, matplotlib, seaborn, scikit-learn
Visualization	matplotlib, seaborn
Models	RandomForest, LogisticRegression, DecisionTree, SVC

## 🚀 How to Run

### Clone this repository:

git clone https://github.com/<your-username>/Loan-Approval-ML.git


### Navigate to the project directory:

cd Loan-Approval-ML


### Install dependencies:

pip install -r requirements.txt


### Run the script:

python loan_approval.py


(Optional) Add loan_data.csv in the same directory.

## 📊 Results Summary
| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Random Forest       | ~0.92    | ~0.91     | ~0.90  | ~0.91    |  |
| Decision Tree       | ~0.86    | ~0.85     | ~0.84  | ~0.84    |
| SVM                 | ~0.82    | ~0.80     | ~0.78  | ~0.79    |

(Note: Actual values depend on dataset variation.)

## 🧠 Future Enhancements

Implement hyperparameter tuning using GridSearchCV

Try XGBoost / LightGBM for improved accuracy

Deploy as a web app using Flask or Streamlit
