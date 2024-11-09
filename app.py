from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib


app = Flask(__name__)

# Load the trained model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get the data
        print(f"Received data: {data}")  # Debugging log
        features = np.array(data['features']).astype(float).reshape(1, -1)

        scaled_input = scaler.transform(features)

        prediction = model.predict(scaled_input)
        
        # Convert the prediction to a Python native float
        prediction_value = float(prediction[0])
        
        return jsonify({'prediction': prediction_value})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")  # Debugging log
        return jsonify({'error': 'An unexpected error occurred: ' + str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
