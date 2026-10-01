import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# ---------- Load saved models (created in the notebook) ----------
@st.cache_resource
def load_all():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.joblib"),
        "Decision Tree": joblib.load("models/decision_tree.joblib"),
        "Random Forest": joblib.load("models/random_forest.joblib"),
        "Random Forest (Tuned - GridSearchCV)": joblib.load("models/random_forest_tuned.joblib"),
        "SVM": joblib.load("models/svm.joblib"),
        "k-NN": joblib.load("models/knn.joblib"),
    }
    scaler = joblib.load("models/scaler.joblib")
    columns = joblib.load("models/feature_columns.joblib")
    return models, scaler, columns

models, scaler, feature_columns = load_all()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details and click **Predict**. All trained models will give their prediction.")

# ---------- User input form ----------
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 30)
    embarked = st.selectbox("Port of embarkation", ["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"])
with col2:
    sibsp = st.slider("Siblings / spouses aboard", 0, 8, 0)
    parch = st.slider("Parents / children aboard", 0, 6, 0)
    fare = st.slider("Fare paid", 0.0, 520.0, 32.0)

model_choice = st.selectbox("Show result for", ["All models"] + list(models.keys()))

# ---------- Prediction ----------
if st.button("Predict"):
    emb = embarked[0]
    row = {
        "Pclass": pclass,
        "Sex": 1 if sex == "female" else 0,   # same encoding as the notebook
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked_Q": 1 if emb == "Q" else 0,
        "Embarked_S": 1 if emb == "S" else 0,
    }
    X = pd.DataFrame([row])[feature_columns]   # same column order as training
    X_scaled = scaler.transform(X)

    selected = models if model_choice == "All models" else {model_choice: models[model_choice]}
    results = []
    for name, m in selected.items():
        pred = m.predict(X_scaled)[0]
        prob = m.predict_proba(X_scaled)[0][1]
        results.append({"Model": name,
                        "Prediction": "Survived ✅" if pred == 1 else "Did not survive ❌",
                        "Survival probability": f"{prob:.1%}"})
        if len(selected) == 1:
            (st.success if pred == 1 else st.error)(f"{name}: {results[-1]['Prediction']} (probability {prob:.1%})")

    if len(selected) > 1:
        st.table(pd.DataFrame(results).set_index("Model"))
        votes = sum(1 for r in results if "Survived ✅" in r["Prediction"])
        st.info(f"{votes} out of {len(results)} models predict survival.")
