from flask import Flask, request, jsonify
"""pip install flask """
import pickle
import numpy as np

with open("uptor_203_linear_model.pkl", "rb") as obj:
    model = pickle.load(obj)


app = Flask(__name__)

@app.route('/')
def landing_page():
    return "Hello welcome to uptor 203 DS"

@app.route('/login',methods=['get'])
def login_page():
    return "Welcome to Login page"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    year = data.get("year")
    if not data or "year" not in data:
        return jsonify({"error": "Please provide JSON body with key 'year'"}), 400
    if isinstance(year, (int, float)):
        year = [year]
    x_value = np.array(year).reshape(-1,1)
    prediction = model.predict(x_value).tolist()

    return jsonify({
        "input": x_value,
        "prediction": prediction
    })

""""
var = {"year":[2000,2001]} ----> Dictionary 
var = '{"year":[2000,2001]}' ----> JSON (String of Dictonary)
"""


if __name__ == "__main__":
    app.run(debug=True)