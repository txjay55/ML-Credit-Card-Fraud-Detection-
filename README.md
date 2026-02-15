# ML-Credit-Card-Fraud-Detection-


# 💳 Credit Card Fraud Detection System

### 🔎 Machine Learning Powered Fraud Risk Monitoring

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions using a highly imbalanced real-world dataset. The solution includes data preprocessing, model building, evaluation, threshold tuning, and deployment via an interactive Streamlit dashboard.

---

## 🚀 Project Overview

Credit card fraud detection is a critical challenge in the financial industry due to:

- Extremely imbalanced data (fraud < 1%)
- High cost of false negatives (missed fraud)
- Need for precision–recall trade-offs

This project builds ML models to classify transactions as:

- ✅ Normal
- 🚨 Fraudulent

---

## 📊 Dataset

**Source:** Kaggle – Credit Card Fraud Detection Dataset  

**Characteristics:**

- 284,807 transactions
- 492 fraud cases (0.17%)
- 30 numerical features
- PCA-transformed features (V1–V28)
- Additional features: `Time`, `Amount`
- Target variable: `Class`

---

## 🧠 Machine Learning Pipeline

### ✔ Data Preprocessing
- Null value verification
- Class imbalance analysis
- Feature scaling (`Amount`)

### ✔ Exploratory Data Analysis (EDA)
- Class distribution visualization
- Amount distribution histogram

### ✔ Models Used
- Logistic Regression (Baseline)
- Random Forest Classifier (Main Model)

### ✔ Evaluation Metrics
Due to class imbalance:

- Precision
- Recall
- F1-score
- ROC-AUC Score

---

## 🎯 Key Results

| Model | Fraud Precision | Fraud Recall | ROC-AUC |
|------|----------------|-------------|---------|
| Logistic Regression | Low | High | ~0.96 |
| Random Forest | **High (0.96)** | Good (~0.80) | ~0.91 |

---

## ⚖ Threshold Tuning

Default threshold (0.5) was adjusted to optimize fraud recall.

**Final Threshold Selected:** `0.3`

✔ Improved fraud detection  
✔ Balanced precision–recall trade-off  

---

## 🌐 Deployment

The model was deployed using **Streamlit** with:

- Interactive input fields
- Fraud prediction output
- Fraud probability gauge meter
- KPI dashboard
- Demo fraud/normal scenarios

---

## 🖥️ Application Features

✔ Transaction analysis interface  
✔ Fraud risk dashboard  
✔ Fraud probability visualization  
✔ Model insights tab  

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Plotly
- Streamlit
- Joblib

---

## 📌 Key Learnings

- Handling imbalanced datasets
- Why accuracy is misleading
- Precision vs Recall trade-offs
- ROC-AUC interpretation
- Threshold tuning strategies
- ML model deployment

---
## 📷 Demo
<img width="1914" height="915" alt="image" src="https://github.com/user-attachments/assets/3fcdc339-e930-48ae-b970-20fb191ae7c6" />


## 🔗 GitHub Repository

https://github.com/txjay55/ML-Credit-Card-Fraud-Detection-.git


---

## 👨‍💻 Author

**Jay keshvala**  
Machine Learning & Data Science Enthusiast

---

## ⭐ If you found this project interesting

Consider giving it a ⭐ on GitHub!
