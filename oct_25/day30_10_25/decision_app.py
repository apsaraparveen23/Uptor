# Apsara_app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained Decision Tree model
with open("uptor_203_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "🌳 Welcome to Uptor 203 Decision Tree API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year = data.get('year')

    if not year:
        return jsonify({"error": "Please provide JSON body with key 'year'"}), 400

    # Handle both single values and lists
    if isinstance(year, (int, float)):
        year = [year]

    x_value = np.array(year).reshape(-1, 1)
    prediction = model.predict(x_value).tolist()

    return jsonify({
        "input": year,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)