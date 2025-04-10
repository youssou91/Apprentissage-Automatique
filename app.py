import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="Projet IA1", layout="wide")
st.title("Projet IA1 : Application d'Apprentissage Automatique")

# Définition des modèles
models_classification = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Arbre de Décision": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

models_regression = {
    "Random Forest": RandomForestRegressor(),
    "Régression Linéaire": LinearRegression(),
    "Arbre de Décision": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor()
}

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Onglets
tab1, tab2 = st.tabs(["Modélisation", "Prédiction"])

# Variables globales
trained_models = {}
feature_names = []

with tab1:
    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader("Téléversez un fichier CSV", type="csv")

    if uploaded_file:
        with st.spinner("Chargement des données..."):
            data = load_data(uploaded_file)
            st.subheader("Aperçu des données")
            st.dataframe(data.head())

            task = st.sidebar.selectbox("Type de tâche", ["Classification", "Régression"])
            target = st.sidebar.selectbox("Variable cible", data.columns)

            if target:
                X = data.drop(columns=[target])
                y = data[target]

                # Sélection des variables numériques pour X
                X_numeric = X.select_dtypes(include=["int64", "float64"])
                if X_numeric.empty:
                    st.error("Aucune colonne numérique trouvée pour entraîner les modèles.")
                else:
                    # Imputation des données pour X
                    imputer_x = SimpleImputer(strategy="most_frequent" if task=="Classification" else "mean")
                    X_imputed = imputer_x.fit_transform(X_numeric)
                    
                    # Pour y, on utilise une stratégie adaptée
                    if task == "Classification":
                        # Vérifier si y est déjà de type string/catégorique.
                        # Si y est numérique mais que l'utilisateur a choisi classification, 
                        # tenter de convertir en entier (si les valeurs discrètes le permettent)
                        if np.issubdtype(y.dtype, np.number):
                            uniques = np.unique(y)
                            # Si le nombre de classes est raisonnable et que toutes les valeurs sont entières après arrondi, on convertit
                            if len(uniques) < 100 and np.all(np.mod(uniques, 1) == 0):
                                y_imputed = y.astype(int).values
                            else:
                                st.error("Pour une tâche de classification, la variable cible doit contenir des catégories discrètes.")
                                st.stop()
                        else:
                            y_imputed = y.astype(str).values
                    else:
                        # Pour la régression, on impute en utilisant la moyenne et on reste en float
                        imputer_y = SimpleImputer(strategy="mean")
                        y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()

                    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.3, random_state=42)

                    metrics_table = []

                    with st.spinner("🔁 Entraînement des modèles..."):
                        if task == "Classification":
                            st.subheader("Résultats - Classification")
                            for name, model in models_classification.items():
                                try:
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    trained_models[name] = model
                                    metrics_table.append({
                                        "Modèle": name,
                                        "Exactitude": round(accuracy_score(y_test, y_pred), 2),
                                        "Précision": round(precision_score(y_test, y_pred, average='weighted'), 2),
                                        "Rappel": round(recall_score(y_test, y_pred, average='weighted'), 2),
                                        "F1-score": round(f1_score(y_test, y_pred, average='weighted'), 2)
                                    })
                                except Exception as e:
                                    st.error(f"Erreur lors de l'entraînement du modèle {name} : {e}")
                        else:
                            st.subheader("Résultats - Régression")
                            for name, model in models_regression.items():
                                try:
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    trained_models[name] = model
                                    metrics_table.append({
                                        "Modèle": name,
                                        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
                                        "MSE": round(mean_squared_error(y_test, y_pred), 2),
                                        "R²": round(r2_score(y_test, y_pred), 2)
                                    })
                                except Exception as e:
                                    st.error(f"Erreur lors de l'entraînement du modèle {name} : {e}")

                    st.subheader("Tableau comparatif des performances")
                    st.table(pd.DataFrame(metrics_table))
                    feature_names.clear()
                    feature_names.extend(X_numeric.columns)

with tab2:
    st.header("Prédiction")
    if not trained_models:
        st.info("Veuillez d'abord entraîner les modèles dans l'onglet 'Modélisation'.")
    else:
        prediction_type = st.selectbox("Type de modèle", ["Classification", "Régression"])
        available_models = models_classification if prediction_type == "Classification" else models_regression
        model_name = st.selectbox("Choisissez un modèle", list(available_models.keys()))
        selected_model = trained_models.get(model_name)

        st.markdown("**Entrez les données pour prédire :**")
        input_data = []
        for feature in feature_names:
            value = st.number_input(f"{feature}", step=1.0, format="%.2f")
            input_data.append(value)

        if st.button("Prédire"):
            try:
                # Les données d'entrée doivent être au format 2D : [ [val1, val2, ...] ]
                pred = selected_model.predict([input_data])
                st.success(f"✅ Résultat de la prédiction : {pred[0]}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
