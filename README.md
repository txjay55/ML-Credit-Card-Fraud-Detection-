# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🔎 Machine Learning Powered Fraud Risk Monitoring Dashboard

An end-to-end Machine Learning project that detects fraudulent credit card transactions using a highly imbalanced real-world dataset. The solution includes data preprocessing, exploratory data analysis, model building, evaluation using proper metrics, threshold tuning, and deployment via an interactive Streamlit dashboard.

---

## 🚀 Project Overview

Credit card fraud detection is a challenging classification problem due to:

- Extremely imbalanced dataset (fraud < 1%)
- High cost of false negatives (missed fraud)
- Need for precision–recall optimization

This system classifies transactions as:

- ✅ Normal Transaction  
- 🚨 Fraudulent Transaction  

---

## 📊 Dataset Information

**Source:** Kaggle – Credit Card Fraud Detection Dataset  

**Dataset Characteristics:**

- 284,807 transactions
- 492 fraud cases (0.17%)
- 30 numerical features
- PCA-transformed anonymized features (`V1 – V28`)
- Additional features: `Time`, `Amount`
- Target variable: `Class`

| Class | Meaning |
|------|---------|
| 0 | Normal |
| 1 | Fraud |

---

## 🧠 Machine Learning Pipeline

### ✔ Data Preprocessing
- Verified null values
- Analyzed class imbalance
- Feature scaling (`Amount`)

### ✔ Exploratory Data Analysis (EDA)
- Class distribution visualization
- Transaction amount distribution

### ✔ Models Implemented
- Logistic Regression (Baseline)
- Random Forest Classifier (Main Model)

### ✔ Evaluation Metrics
Since accuracy is misleading for imbalanced data:

- Precision
- Recall
- F1-score
- ROC-AUC Score

---

## 🎯 Key Results

| Model | Fraud Precision | Fraud Recall | ROC-AUC |
|------|----------------|-------------|---------|
| Logistic Regression | Low | High | ~0.96 |
| Random Forest | **0.96** | **~0.80** | **~0.91** |

---

## ⚖ Threshold Tuning

To improve fraud detection performance:

- Adjusted default threshold (0.5)
- Tested multiple thresholds
- Selected optimal threshold

**Final Threshold Chosen:** `0.3`

✔ Improved fraud recall  
✔ Balanced precision–recall trade-off  

---

## 🌐 Deployment

The trained Random Forest model was deployed using **Streamlit**.

### 🖥️ Application Features

✔ Transaction Analysis Interface  
✔ Fraud Risk Dashboard  
✔ Fraud Probability Gauge Meter  
✔ KPI Metrics Display  
✔ Demo Fraud / Normal Scenarios  
✔ Model Insights Tab  

---

## 📷 Demo



<img width="1914" height="915" alt="image" src="https://github.com/user-attachments/assets/3fcdc339-e930-48ae-b970-20fb191ae7c6" />

---

## 🛠️ Tech Stack

**Languages & Libraries:**

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib

---

## 📌 Key Learnings

- Handling imbalanced datasets
- Why accuracy is unreliable
- Precision vs Recall trade-offs
- ROC-AUC interpretation
- Threshold tuning strategies
- ML model deployment
- Dashboard UI design

---

## 🔗 GitHub Repository

👉 https://github.com/txjay55/ML-Credit-Card-Fraud-Detection-.git

---

## 👨‍💻 Author

**Jay**  
Machine Learning & Data Science Enthusiast 🚀

---

## ⭐ Support

If you found this project interesting, consider giving it a ⭐ on GitHub!












