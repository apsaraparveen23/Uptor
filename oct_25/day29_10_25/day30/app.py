from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

uptor_app = Flask(__name__)


@uptor_app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Welcome to Linear Regression API"
    })


@uptor_app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "year" not in data:
        return jsonify({"error": "Please provide JSON body with key 'year'"}), 400

    x_value = float(data["year"])
    prediction = model.predict(np.array([[x_value]]))[0]

    return jsonify({
        "input": x_value,
        "prediction": prediction
    })


if __name__ == "__main__":
    uptor_app.run(debug=True)
