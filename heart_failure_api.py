from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load

# Load your trained model
model = load("decision_tree_model.joblib")

X = pd.read_csv("Thyroid_Dataset_Resampled.csv")

# Initialize app
api = Flask(__name__)
CORS(api)  # Enables CORS globally (still add @cross_origin to be safe)

@api.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin(origins='*')  # Allow requests from all origins
def predict_thyroid_risk():
    if request.method == 'OPTIONS':
        # CORS preflight request
        return '', 200

    try:
        data = request.json['inputs']
        input_df = pd.DataFrame(data)

        # Predict
        probabilities = model.predict_proba(input_df)[0]
        risk_levels = {
            "No Risk": probabilities[0] * 100,
            "Moderate Risk": probabilities[1] * 100,
            "High Risk": probabilities[2] * 100
        }

        predicted_class = max(risk_levels, key=risk_levels.get)
        suggestions = {
            "No Risk": "You currently show no signs of thyroid risk. Maintain a balanced lifestyle and regular health check-ups.",
            "Moderate Risk": "There are moderate indicators of thyroid issues. Consider scheduling a thyroid function test.",
            "High Risk": "There is a high risk of thyroid dysfunction. Please consult a healthcare provider as soon as possible for a full diagnosis."
        }

        return jsonify({
            "prediction": predicted_class,
            "risk_percentages": risk_levels,
            "suggestion": suggestions[predicted_class]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    api.run(debug=True, host='0.0.0.0', port=5000)
