from utilities.file_reading import data_to_data_frame
from machine_learning_linear_regression import model_training
from sklearn.linear_model import LinearRegression

my_input={'year':[2009,2010]}
input_df=data_to_data_frame(my_input)

def final_prediction():
    model=model_training(LinearRegression())
    my_prediction=model.predict(input_df)
    print("my final output")
    print(my_prediction)
    print("**************")

if __name__ == "__main__":
    final_prediction()

