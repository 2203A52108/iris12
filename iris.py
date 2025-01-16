import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Custom CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            font-family: 'Poppins', sans-serif;
            color: white;
        }
        .header {
            text-align: center;
            padding: 10px;
            font-size: 36px;
            font-weight: bold;
            color: white;
            background-color: #ff6f61;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .section {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .button {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .button:hover {
            background: linear-gradient(to right, #2575fc, #6a11cb);
        }
    </style>
""", unsafe_allow_html=True)

# App title with styled header
st.markdown("<div class='header'>Iris Dataset Real-Time Analysis and Classification</div>", unsafe_allow_html=True)

# Upload dataset
uploaded_file = st.file_uploader("Upload Iris Dataset CSV", type=["csv"], help="Upload the Iris dataset in CSV format.")
if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.markdown("<div class='section'><h3>Dataset Preview</h3></div>", unsafe_allow_html=True)
    st.write(df.head())

    # Display dataset summary
    st.markdown("<div class='section'><h3>Dataset Summary</h3></div>", unsafe_allow_html=True)
    st.write(df.describe())

    # Display dataset shape
    st.markdown("<div class='section'><h3>Dataset Shape</h3></div>", unsafe_allow_html=True)
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Species distribution
    st.markdown("<div class='section'><h3>Species Distribution</h3></div>", unsafe_allow_html=True)
    species_counts = df["Species"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    species_counts.plot(kind="bar", color="skyblue", ax=ax)
    plt.title("Species Distribution")
    plt.xlabel("Species")
    plt.ylabel("Count")
    st.pyplot(fig)

    # Preprocess the data
    X = df.drop(columns=["Species"])
    y = LabelEncoder().fit_transform(df["Species"])  # Convert species to numeric values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVM Classification
    if st.button("Run SVM Model", key="svm", help="Click to run the SVM model"):
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        svm_y_pred = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_y_pred)
        st.markdown("<div class='section'><h3>SVM Results</h3></div>", unsafe_allow_html=True)
        st.write(f"### SVM Accuracy: {svm_accuracy}")
        st.write("#### SVM Classification Report")
        st.text(classification_report(y_test, svm_y_pred))

    # Perceptron Classification
    if st.button("Run Perceptron Model", key="perceptron", help="Click to run the Perceptron model"):
        perceptron_model = Perceptron()
        perceptron_model.fit(X_train, y_train)
        perceptron_y_pred = perceptron_model.predict(X_test)
        perceptron_accuracy = accuracy_score(y_test, perceptron_y_pred)
        st.markdown("<div class='section'><h3>Perceptron Results</h3></div>", unsafe_allow_html=True)
        st.write(f"### Perceptron Accuracy: {perceptron_accuracy}")
        st.write("#### Perceptron Classification Report")
        st.text(classification_report(y_test, perceptron_y_pred))

    # Logistic Regression Custom Implementation
    if st.button("Run Logistic Regression Model", key="logistic", help="Click to run the Logistic Regression model"):
        class LogisticRegression:
            def __init__(self, learning_rate=0.01, num_iterations=1000):
                self.learning_rate = learning_rate
                self.num_iterations = num_iterations
                self.weights = None
                self.bias = None

            def sigmoid(self, z):
                return 1 / (1 + np.exp(-z))

            def fit(self, X, Y):
                m, n = X.shape
                self.weights = np.zeros(n)
                self.bias = 0
                for _ in range(self.num_iterations):
                    h = self.sigmoid(np.dot(X, self.weights) + self.bias)
                    dw = (1 / m) * np.dot(X.T, (h - Y))
                    db = (1 / m) * np.sum(h - Y)
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db

            def predict(self, X):
                h = self.sigmoid(np.dot(X, self.weights) + self.bias)
                return np.where(h > 0.5, 1, 0)

        # Prepare data
        binary_y = (df["Species"] == "Iris-versicolor").astype(int)  # Binary classification
        X = df.drop(columns=["Species"])
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias column
        X_train, X_test, y_train, y_test = train_test_split(X, binary_y, test_size=0.2, random_state=42)

        # Logistic Regression Model
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        lr_y_pred = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_y_pred)
        st.markdown("<div class='section'><h3>Logistic Regression Results</h3></div>", unsafe_allow_html=True)
        st.write(f"### Logistic Regression Accuracy: {lr_accuracy}")
        st.write("#### Logistic Regression Classification Report")
        st.text(classification_report(y_test, lr_y_pred))
