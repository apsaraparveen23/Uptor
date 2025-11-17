from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return "✅ Iris Prediction Model is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON input: {"data": [5.1, 3.5, 1.4, 0.2]}
        data = request.get_json(force=True)
        input_data = np.array(data['data']).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
