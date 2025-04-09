import streamlit as st
import joblib
import numpy as np

# Charger le modèle et les encodeurs
model = joblib.load("modele.pkl")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Prédiction d'emploi", page_icon="🔮")
st.title("🔍 Prédiction du type d'emploi")

st.markdown("Remplis les informations ci-dessous pour prédire le type d'emploi 👷‍♂️👩‍💼")

# Interface utilisateur
compte_bancaire = st.selectbox("🏦 Compte bancaire :", label_encoders["compte_bancaire"].classes_)
type_localisation = st.selectbox("📍 Type de localisation :", label_encoders["type_localisation"].classes_)
accès_téléphone = st.selectbox("📱 Accès au téléphone :", label_encoders["accès_téléphone"].classes_)
taille_foyer = st.number_input("👨‍👩‍👧‍👦 Taille du foyer :", min_value=1, max_value=20)
âge = st.number_input("🎂 Âge :", min_value=10, max_value=100)
sexe = st.selectbox("⚧ Sexe :", label_encoders["sexe"].classes_)
statut_matrimonial = st.selectbox("💍 Statut matrimonial :", label_encoders["statut_matrimonial"].classes_)
niveau_éducation = st.selectbox("🎓 Niveau d'éducation :", label_encoders["niveau_éducation"].classes_)

# Encodage des valeurs saisies
input_data = np.array([[
    label_encoders["compte_bancaire"].transform([compte_bancaire])[0],
    label_encoders["type_localisation"].transform([type_localisation])[0],
    label_encoders["accès_téléphone"].transform([accès_téléphone])[0],
    taille_foyer,
    âge,
    label_encoders["sexe"].transform([sexe])[0],
    label_encoders["statut_matrimonial"].transform([statut_matrimonial])[0],
    label_encoders["niveau_éducation"].transform([niveau_éducation])[0]
]], dtype=np.float32)

# Bouton de prédiction
if st.button("🔮 Prédire le type d'emploi"):
    prediction = model.predict(input_data)[0]
    st.success(f"🧠 Le type d'emploi prédit est : **{prediction}**")
