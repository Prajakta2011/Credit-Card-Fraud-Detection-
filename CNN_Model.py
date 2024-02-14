import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up the Streamlit app
st.title("Credit Card Fraud Detection")

# Load the credit card dataset
dataset_path = "credit_card_dataset.csv"
df = pd.read_csv('creditcard.csv')

# Split the dataset into features and labels
X = df.drop("Class", axis=1).values
y = df["Class"].values

# Perform feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Determine the image dimensions based on the number of features
num_features = X.shape[1]
image_width = int(np.sqrt(num_features))
image_height = int(np.ceil(num_features / image_width))
num_channels = 1

# Reshape the features to image-like representations
X = X.reshape(-1, image_width, image_height, num_channels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, num_channels)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Function to preprocess the input features
def preprocess_features(features):
    # Perform feature scaling on the input features
    scaled_features = scaler.transform(features)

    # Reshape the features to match the input shape of the CNN model
    reshaped_features = scaled_features.reshape(-1, image_width, image_height, num_channels)

    # Return the preprocessed features
    return reshaped_features

# Function to predict fraud using the trained CNN model
def predict_fraud(features):
    # Preprocess the input features
    processed_features = preprocess_features(features)

    # Perform the prediction
    prediction = model.predict(processed_features)

    # Return the prediction
    return prediction

# Main app logic
def main():
    # Create input form for credit card transaction details
    amount = st.number_input("Amount", value=0.0, step=0.01)
    time = st.number_input("Time", value=0.0, step=1.0)
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    # Include other relevant features in a similar manner

    if st.button("Predict"):
        # Create a feature array from the input values
        features = [[amount, time, v1, v2]]  # Add other relevant features

        # Perform the fraud prediction
        prediction = predict_fraud(features)

        # Determine the class label based on the prediction
        if prediction[0][0] > 0.5:
            st.write("Fraudulent Transaction")
        else:
            st.write("Legitimate Transaction")

# Run the app
if __name__== "_main_":
    main()