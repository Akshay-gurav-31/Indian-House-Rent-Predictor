from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# -------------------- Load Model -------------------- #
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model", "model.pkl")

try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except Exception as err:
    print(f"Failed to load model: {err}")
    model = None


# -------------------- Routes -------------------- #
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({
            "error": "Model not available. Train or provide the model first."
        }), 500

    try:
        input_data = request.get_json()
        print("Input received:", input_data)

        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        result = model.predict(input_df)

        # Prevent negative price
        price = max(0, float(result[0]))

        return jsonify({
            "success": True,
            "predicted_price": round(price, 2)
        })

    except Exception as err:
        print("Error during prediction:", err)
        return jsonify({
            "error": str(err)
        }), 400


# -------------------- Run App -------------------- #
if __name__ == "__main__":
    app.run(debug=True, port=5000)
