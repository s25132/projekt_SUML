import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os
import pickle


# --- Ensure output dirs exist ---
os.makedirs("results", exist_ok=True)
os.makedirs("app/model", exist_ok=True)

# Dane przykładowe
df = pd.read_csv('data/Titanic.csv')

colums_to_remove = ['home.dest', 'boat', 'ticket', 'parch', 'sibsp', 'name', 'body']

# Usuń je, jeśli istnieją (bezpiecznie)
df = df.drop(columns=[col for col in colums_to_remove if col in df.columns])

X = df.drop(columns=['survived'])
y = df['survived']

# Zamiana danych typu object na kolumny liczbowe
X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded.columns = X_encoded.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

# Podział
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Model LightGBM
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

with open("app/model/lightgbm_model.pkl", "wb") as f:
    pickle.dump(model, f)


# Predykcja
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # prawdopodobieństwo dla klasy 1

# 5. Ewaluacja
print("=== Ewaluacja ===")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))


# === 6. SHAP – wyjaśnienie jednej predykcji ===
explainer = shap.Explainer(model)

# Wybierz jedną próbkę do wyjaśnienia
sample = X_test.iloc[[0]]
shap_values = explainer(sample)

# === 7. Wyświetlenie waterfall plot ===
print("Predykcja:", model.predict(sample)[0])
print("Prawdopodobieństwo:", model.predict_proba(sample)[0])

# Zakładamy, że masz już obiekt shap_values
shap.plots.waterfall(shap_values[0], show=False)

# Zapisz wykres do pliku (np. PNG)
plt.savefig("results/shap_waterfall.png", bbox_inches='tight')
plt.close()
