import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


column_names = ['buying_price', 'maintenance_cost', 'doors_numbers',
                'person_number', 'luggage_space', 'safety', 'class']
df = pd.read_csv("car_evaluation.csv", names=column_names, header=None)


label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])
X = df.drop("class", axis=1)
encoder = ce.OrdinalEncoder(cols=['buying_price', 'maintenance_cost', 'doors_numbers',
                                  'person_number', 'luggage_space', 'safety'])
X_encoded = encoder.fit_transform(X)


x = X_encoded
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)

accuracy_checkout_output = accuracy_score(y_test, y_pred)
print(accuracy_checkout_output)

"""Problem statement
train_test_split()
this function basically takes 70% smaple for training and 30% for tetsing.

problem
sometime  test data which is 30% may have better data that produces the output than the testing data
which is 70% is nature

Resolution:
Cross_validation--Ensure you apply the model x data,y data and metrics find out average

"""
