# 🛡️ Ethereum Fraud Detection

ML-powered system لكشف حسابات Ethereum المحتالة بناءً على سلوك المعاملات.

> Final Project - **Epsilon AI Data Science Track**
> Main Repo: [Epsilon AI](https://github.com/EpsilonAI)

---

## 🎯 Problem Statement

في عالم العملات الرقمية، فيه محتالين بيستخدموا حسابات وهمية للاحتيال على الناس وسرقة الـ Ether.
المشروع ده بيستخدم Machine Learning لكشف الحسابات المشبوهة قبل ما تعمل ضرر.

---

## 📊 Dataset

- **Source:** [Kaggle - Ethereum Fraud Detection](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
- **Size:** 9,841 accounts × 51 features
- **Target:** `FLAG` (0 = Normal, 1 = Fraud)
- **Imbalance:** 22.14% fraud

---

## 🔧 Pipeline

| Step | Description |
|---|---|
| 1️⃣ Data Cleaning | Handle missing values, drop constant/duplicate columns |
| 2️⃣ EDA | 8+ visualizations + statistical tests (T-test, correlations) |
| 3️⃣ Feature Engineering | 7 new features (ratios, activity scores) |
| 4️⃣ Feature Selection | Filter + Wrapper + Embedded methods → 20 features |
| 5️⃣ Modeling | 5 algorithms compared (LR, DT, RF, XGB, KNN) |
| 6️⃣ Tuning | GridSearchCV on XGBoost (270 fits) |
| 7️⃣ Validation | 5-Fold Stratified Cross-Validation |
| 8️⃣ Deployment | Streamlit web app |

---

## 🏆 Best Model: XGBoost (Tuned)

| Metric | Value |
|---|---|
| Accuracy | 98.88% |
| Precision | 98.36% |
| **Recall** | **96.56%** |
| **F1-Score** | **97.45%** |
| ROC-AUC | 99.77% |

**Best Hyperparameters:**
- `n_estimators`: 300
- `max_depth`: 5
- `learning_rate`: 0.1
- `subsample`: 0.8

---

## 🚀 Live Demo

🔗 **[Try the app here](#)** ← (هنحط الـ link بعد deployment)

---

## 📁 Repository Structure

```
ethereum-fraud-detection/
├── app.py                       # Streamlit application
├── notebook.ipynb               # Full analysis notebook
├── best_model.pkl               # Trained XGBoost model
├── scaler.pkl                   # StandardScaler
├── selected_features.pkl        # 20 selected features
├── requirements.txt             # Dependencies
├── data/
│   └── transaction_dataset.csv  # Original dataset
└── README.md
```

---

## ⚡ Quick Start

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/ethereum-fraud-detection.git
cd ethereum-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

افتح `http://localhost:8501` في الـ browser.

---

## 🛠️ Tech Stack

`Python 3.10` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Imbalanced-learn` `Streamlit` `Matplotlib` `Seaborn`

---

## 📊 Key Insights

1. **`has_erc20_rec_token`** هو أقوى مؤشر للـ fraud (correlation = 0.55)
2. الحسابات المحتالة **عمرها أقصر** بكتير (Fraud Detection Rate: 96.56%)
3. **SMOTE** + **XGBoost** أعطى أفضل نتائج للـ imbalanced data
4. False Alarm Rate < **0.5%**

---

## 👤 Author

- **Name:** [Your Name]
- **Course:** Epsilon AI Data Science
- **Date:** 2026

---

## 📜 License

MIT License - يفضل استخدام المشروع للأغراض التعليمية فقط.
