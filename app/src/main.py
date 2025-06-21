import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# === Wczytaj model ===
model_path = "model/lightgbm_model.pkl"

@st.cache_resource
def load_model():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# === Tytuł ===
st.title("Predykcja przeżycia katastrofy Titanica na podstawie danych pasażera")

# === Formularz danych ===
st.subheader("Wprowadź dane")

pclass = st.selectbox("Klasa (pclass)", [1, 2, 3])
sex = st.selectbox("Płeć (sex)", ["male", "female"])
age = st.number_input("Wiek (age)", min_value=0.0, max_value=100.0, value=30.0)
fare = st.number_input("Cena biletu (fare)", min_value=0.0, value=50.0)
cabin = st.text_input("Kabina (cabin)")
embarked = st.selectbox("Port zaokrętowania (embarked)", ["C", "Q", "S"])

# === Przygotuj dane ===
if st.button("Przewiduj"):
    input_dict = {
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "fare": fare,
        "cabin": cabin,
        "embarked": embarked
    }

    input_df = pd.DataFrame([input_dict])

    # Wstępne przetwarzanie zgodne z modelem (np. get_dummies)
    input_df = pd.get_dummies(input_df)

    # Dopasowanie kolumn do modelu
    model_features = model.feature_name_
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0  # dodaj brakujące kolumny jako 0
    input_df = input_df[model_features]  # zachowaj kolejność kolumn

    # Predykcja
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    label = "Pasażer przeżył" if pred == 1 else "Pasażer nie przeżył"

    st.success(f"Predykcja: {label} (prawdopodobieństwo: {proba:.2f})")

    st.subheader("Wyjaśnienie predykcji (SHAP)")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)


    shap.plots.waterfall(shap_values[0], show=False)

    # Pobierz bieżącą figurę
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()  # Wyczyść po sobie