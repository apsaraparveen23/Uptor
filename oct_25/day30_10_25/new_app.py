from flask import  Flask,request,jsonify
import pickle
import  numpy as np

app = Flask(__name__)
with open("upor_203_linear_model.pkl", "rb") as obj:
 model=pickle.load(obj)




@app.route('/')
def landing_page():
  return "Hello welcome to uptor 203"

@app.route('/login', methods=['GET'])

def login_page():
  return "Hello welcome to Login Page"

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if not data or "year" not in data:
#         return jsonify({"error": "Please provide JSON body with key 'year'"}), 400
#
#     years = data["year"]
#     if isinstance(years, list):
#         predictions = model.predict(np.array(years).reshape(-1, 1)).tolist()
#         return jsonify({
#             "input": years,
#             "predictions": predictions
#         })
#     else:
#         x_value = float(years)
#         prediction = model.predict(np.array([[x_value]]))[0]
#         return jsonify({
#             "input": x_value,
#             "prediction": prediction
#         })


# @app.route('/predict',methods=['POST'])
# def predict():
#     data=request.get_json()
#      year=data.get('year')
#     if not year:
#         return jsonify({"error": "Please provide JSON body with key 'year'"}), 400
#
#     if isinstance(year,(int,float)):
#         year=[year]
#     x_value=np.array(year).reshape(-1,1)
#     prediction=model.predict(x_value).tolist() #returns 0th column,converting to 2D array
#
#     return jsonify({
#         "input":year,
#         "prediction":prediction
#     })

#the above post code also works

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year = data.get('year')  # Fixed indentation

    if not year:
        return jsonify({"error": "Please provide JSON body with key 'year'"}), 400

    # If year is a single number, convert it to a list
    if isinstance(year, (int, float)):
        year = [year]

    # Reshape for model prediction
    x_value = np.array(year).reshape(-1, 1)
    prediction = model.predict(x_value).tolist()

    return jsonify({
        "input": year,
        "prediction": prediction
    })

"""
var={"year":[2000,2001]}----Dictionary
var='{"year":[2000,2001]}'----JSON(String for Dictionary)

"""
if __name__ =="__main__":
 app.run(debug=True)