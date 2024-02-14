import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into features (X) and labels (y)
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Reshape X to 3D for CNN input (assuming features are already normalized)
X = X.values.reshape(-1, 30, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Build CNN model
model = keras.Sequential([
    keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(30, 1)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate model performance
train_loss, train_acc = model.evaluate(X_train, y_train)
test_loss, test_acc = model.evaluate(X_test, y_test)

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')

# Create a button to submit input and get prediction
submit = st.button('Convolutional Neural Network')

if submit:
    # Get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # Reshape features to match CNN input shape
    features = features.reshape(1, 30, 1)
    # Make prediction
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction[0])
    # Display result
    if predicted_class == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")