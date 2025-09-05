import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import category_encoders as ce

from Aug_25.day14_08_25.machine_learning_linear_regression_csv import x_train

column_names = ['buying_price', 'maintenance_cost', 'doors_numbers',
                'person_number', 'luggage_space', 'safety', 'class']
df = pd.read_csv("car_evaluation.csv", names=column_names, header=None)


label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

X = df.drop("class", axis=1)
y = df['class']

encoder = ce.OrdinalEncoder(cols=['buying_price', 'maintenance_cost', 'doors_numbers',
                                  'person_number', 'luggage_space', 'safety'])
X_encoded = encoder.fit_transform(X)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_encoded, y)
y_pred = dt_model.predict(X_encoded)
print(y_pred)

#
# column_names = ['buying_price', 'maintenance_cost', 'doors_numbers',
#                 'person_number', 'luggage_space', 'safety', 'class']
# df = pd.read_csv("car_evaluation.csv", names=column_names, header=None)
#
#
# label_encoder = LabelEncoder()
# df['class'] = label_encoder.fit_transform(df['class'])
#
# X = df.drop("class", axis=1)
# y = df['class']
#
# encoder = ce.OrdinalEncoder(cols=['buying_price', 'maintenance_cost', 'doors_numbers',
#                                   'person_number', 'luggage_space', 'safety'])
# X_encoded = encoder.fit_transform(X)
# print(X_encoded)
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
#
# model= DecisionTreeClassifier(random_state=42)
# model.fit(x_train y_train)
# y_pred = model.predict(X_encoded)
# print(y_pred)
#
#
