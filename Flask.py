from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle et le scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON
        data = request.get_json()
        
        # Créer un DataFrame avec les données reçues
        input_data = pd.DataFrame({
            'area': [data['area']],
            'bedrooms': [data['bedrooms']],
            'bathrooms': [data['bathrooms']],
            'stories': [data['stories']],
            'mainroad': [1 if data['mainroad'] == 'Oui' else 0],
            'guestroom': [1 if data['guestroom'] == 'Oui' else 0],
            'basement': [1 if data['basement'] == 'Oui' else 0],
            'hotwaterheating': [1 if data['hotwaterheating'] == 'Oui' else 0],
            'airconditioning': [1 if data['airconditioning'] == 'Oui' else 0],
            'parking': [data['parking']],
            'prefarea': [1 if data['prefarea'] == 'Oui' else 0],
            'furnishingstatus': [0 if data['furnishingstatus'] == 'Meublé' else 1 if data['furnishingstatus'] == 'Semi-meublé' else 2]
        })
        
        # Appliquer le scaling
        input_scaled = scaler.transform(input_data)
        
        # Faire la prédiction
        prediction = model.predict(input_scaled)
        
        # Retourner le résultat
        return jsonify({'predicted_price': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)