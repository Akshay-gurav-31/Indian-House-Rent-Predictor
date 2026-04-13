from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Load the trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'model.pkl')

try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not trained or found. Please train first.'}), 500
        
    try:
        data = request.json
        print("Received prediction request:", data)
        # Create DataFrame from input data
        # data should be something like:
        # { 'bedroom': 2, 'area': 1200, 'bathroom': 2, 'furnish_type': 'Furnished', 
        #   'city': 'Ahmedabad', 'property_type': 'Apartment', 'seller_type': 'OWNER' }
        df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(df)
        
        return jsonify({
            'success': True,
            'predicted_price': round(prediction[0], 2)
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
