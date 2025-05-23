import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load model
model = load("decision_tree_model.joblib")

# Load and inspect training data
X = pd.read_csv("Thyroid_Dataset_Resampled.csv")

api = Flask(__name__)
CORS(api)

@api.route('/api/doa_prediction', methods=['POST'])
def predict_thyroid_risk():
    data = request.json['inputs']
    input_df = pd.DataFrame(data)

    # Predict class probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Map output to percentage for each risk level
    risk_levels = {
        "No Risk": probabilities[0] * 100,        # Assuming class 0 is No Risk
        "Moderate Risk": probabilities[1] * 100,  # Assuming class 1 is Moderate Risk
        "High Risk": probabilities[2] * 100       # Assuming class 2 is High Risk
    }

    # Get predicted class (the one with the highest probability)
    predicted_class = max(risk_levels, key=risk_levels.get)

    # Add personalized suggestions
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

if __name__ == "__main__":
    api.run(debug=True, host='0.0.0.0')
