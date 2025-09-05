from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix


#load the iris dataset as an example
afsara=load_iris()
X=afsara.data
y=afsara.target

#split the data in to training and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)#random_state=0 ensures reproducibility—same split every time you run it.


#fit a decision tree model to the training data
tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)

"""Evaluating the performance of a Decision Tree model
Once a decision tree model has been fit to data,we need to evaluate its performance to determine

**Accuracy**The fraction of correct predictions made by a model.

**Confusion matrix** A table that summarizes the number of true positives,true negative,false positive and false negative
predictions made by the model.

**Precision** The fraction of positive predictions that are actually positive.

**Recall** The fraction of actual positive cases that were correctly predicted as positive by the model

**F1 score** The harmonic mean of precision and recall.

**classification_report**
"""

#make predictions on the test set
y_pred=tree.predict(X_test)

#calculate the accuracy ,precision,recall and F1-score
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='weighted')
recall=recall_score(y_test,y_pred,average='weighted')
f1=f1_score(y_test,y_pred,average='weighted')
confusion_matrix=confusion_matrix(y_test,y_pred)

#print the results
print(f"Accuracy:{accuracy}")
print("Precision:",precision)
# print("Recall:" % recall)
# print("F1-score:" % f1)
print("Recall:", recall)
print("F1-score:",f1)
print(confusion_matrix)