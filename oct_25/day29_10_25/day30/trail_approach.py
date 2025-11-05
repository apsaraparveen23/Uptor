from flask import Flask, request, jsonify
import pandas as pd
import pickle

application = Flask(__name__)

with open("linear_model.pkl", "rb") as obj:
    model = pickle.load(obj)

@application.route("/predict", method=['POST'])
def price_prediction():
    data = request.get_json()

    if not data or "year" not in data:
        return jsonify({"error": "Please provide JSON body with key 'year'"}), 400

    # Convert input to DataFrame
    df_input = pd.DataFrame({"year": data["year"]})

    # Predict
    predictions = model.predict(df_input)

    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    application.run(debug=True)