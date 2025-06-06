# 🏠 House Price Prediction Model

A machine learning model that predicts house prices based on various features like area, bedrooms, bathrooms, and other amenities.



## 📌 Features

- **Accurate Predictions**: Trained on comprehensive housing data
- **Multiple Models**: Includes Linear Regression, Random Forest, and SVM
- **Web Interface**: Streamlit app for easy interaction
- **API Endpoint**: Flask integration for developers
- **Detailed Analysis**: Complete EDA and model evaluation

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/farahoumezzine/House-Price-Prediction.git


Usage:
 streamlit run app.py

Flask API:
python Flask.py
     curl -X POST -H "Content-Type: application/json" -d '{
       "area": 1500,
       "bedrooms": 3,
       "bathrooms": 2,
       "stories": 2,
       "mainroad": "Oui",
       "guestroom": "Non",
       "basement": "Oui",
       "hotwaterheating": "Non",
       "airconditioning": "Oui",
       "parking": 1,
       "prefarea": "Oui",
       "furnishingstatus": "Meublé"
     }' http://127.0.0.1:5000/predict


Test the entire workflow:
Run dataprep.py to verify data loading.
Run datapreproce.py to verify preprocessing.
Run models.py to verify model training.
Run optimize.py to verify model optimization.
Test the Streamlit app and Flask API to verify predictions.