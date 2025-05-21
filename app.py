import streamlit as st
import pandas as pd
import joblib

# Charger le modèle sauvegardé
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Prédiction du Prix des Maisons')

# Formulaire de saisie
st.header('Entrez les caractéristiques de la maison')

area = st.number_input('Surface (sqft)', min_value=500, max_value=20000, value=1500)
bedrooms = st.number_input('Nombre de chambres', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Nombre de salles de bain', min_value=1, max_value=5, value=2)
stories = st.number_input('Nombre d\'étages', min_value=1, max_value=4, value=2)
mainroad = st.selectbox('Proche d\'une route principale', ['Oui', 'Non'])
guestroom = st.selectbox('Avec chambre d\'amis', ['Oui', 'Non'])
basement = st.selectbox('Avec sous-sol', ['Oui', 'Non'])
hotwaterheating = st.selectbox('Chauffage à eau chaude', ['Oui', 'Non'])
airconditioning = st.selectbox('Climatisation', ['Oui', 'Non'])
parking = st.number_input('Nombre de places de parking', min_value=0, max_value=4, value=1)
prefarea = st.selectbox('Zone préférentielle', ['Oui', 'Non'])
furnishingstatus = st.selectbox('État des meubles', ['Meublé', 'Semi-meublé', 'Non meublé'])

# Préparation des données
if st.button('Prédire le prix'):
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [1 if mainroad == 'Oui' else 0],
        'guestroom': [1 if guestroom == 'Oui' else 0],
        'basement': [1 if basement == 'Oui' else 0],
        'hotwaterheating': [1 if hotwaterheating == 'Oui' else 0],
        'airconditioning': [1 if airconditioning == 'Oui' else 0],
        'parking': [parking],
        'prefarea': [1 if prefarea == 'Oui' else 0],
        'furnishingstatus': [0 if furnishingstatus == 'Meublé' else 1 if furnishingstatus == 'Semi-meublé' else 2]
    })
    
    # Scaling et prédiction
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    st.success(f'Prix estimé: {prediction[0]:,.2f} €')