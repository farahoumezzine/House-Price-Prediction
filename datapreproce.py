from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data():
    # Charger les données
    data = pd.read_csv('Housing_Price_Data.csv')
    
    # Encodage des variables catégorielles
    cat_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                'airconditioning', 'prefarea', 'furnishingstatus']
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])
    
    # Séparation des features et de la target
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Division en train/test
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

# Call the function and store the results
(X_train, X_test, y_train, y_test), scaler = preprocess_data()