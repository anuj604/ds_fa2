# 🚢 Titanic Survival Predictor

An interactive Machine Learning web application built with **Streamlit** and **Scikit-learn** to predict passenger survival on the Titanic based on trained ML models.

---

## 🌟 Overview
This project presents an end-to-end Machine Learning pipeline and web interface for the classic Titanic dataset. Multiple machine learning classifiers were trained, tuned, and evaluated in the Jupyter Notebook (`FA2_ML_Case_Study.ipynb`), and saved as production artifacts (`.joblib`) for real-time web deployment.

### 🧠 Trained Models Included:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest** (Default)
- **Random Forest** (Tuned via `GridSearchCV`)
- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (k-NN)**

---

## 📁 Repository Structure
```text
.
├── app.py                   # Streamlit web application
├── FA2_ML_Case_Study.ipynb  # Notebook: EDA, preprocessing, model training & evaluation
├── models/                  # Pre-trained model artifacts & scaler
│   ├── decision_tree.joblib
│   ├── feature_columns.joblib
│   ├── knn.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── random_forest_tuned.joblib
│   ├── scaler.joblib
│   └── svm.joblib
├── train.csv                # Dataset used for training
├── requirements.txt         # Dependencies for Streamlit Cloud deployment
├── .gitignore
└── README.md
```

---

## ⚙️ Local Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/anuj604/ds_fa2.git
cd ds_fa2
```

### 2. Create a virtual environment & install requirements
```bash
python -m venv .venv
# Activate virtual environment:
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Streamlit App
```bash
streamlit run app.py
```

---

## 🚀 Deployment on Streamlit Community Cloud

To deploy this app publicly:
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New App** -> **Use existing repo**.
3. Select your repository (`anuj604/ds_fa2`), branch (`main`), and set Main file path to `app.py`.
4. Click **Deploy!**

---

## 👤 Author
Developed by **Anuj** ([@anuj604](https://github.com/anuj604/))
