# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

#
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

#
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Split the dataset into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create a decision tree classifier and fit it to the training data
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Use the classifier to make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit1 = st.button("Decision Tree")

#LR
# Load the dataset
data = pd.read_csv("creditcard.csv")

# Split the dataset into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train logistic regression model
model1= LogisticRegression()
model1.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model1.predict(X_train), y_train)
test_acc = accuracy_score(model1.predict(X_test), y_test)

submit2 = st.button('Logistic Regression')

#SVM
# Load the dataset
data = pd.read_csv("creditcard.csv")

# Split the dataset into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model2 = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)
submit3 = st.button('SVM')

#Random Forest
# Load the dataset
data = pd.read_csv("creditcard.csv")

# Split the dataset into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model3.fit(X_train, y_train)

# Use the classifier to make predictions on the test data
y_pred = model3.predict(X_test)
submit4 = st.button('Random Forest')



if submit1:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")


elif submit2:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model1.predict(features.reshape(1, -1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")

elif submit3:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model2.predict(features.reshape(1, -1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")

else:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model3.predict(features.reshape(1, -1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")


