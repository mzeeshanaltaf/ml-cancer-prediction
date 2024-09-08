import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)


# This function reads the data from csv file and returns the cleaned after pre-processing
def get_breast_cancer_data():
    # Read breast cancer data from csv file
    data = pd.read_csv("data/breast_cancer_data.csv")

    # Drop unnecessary column(s)
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Perform label encoding. 1 to M (malignant) and 0 to B (benign)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


# Function to train the machine learning model
def train_model(model_name):
    # Available models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gaussian NB": GaussianNB()

    }
    # Get the data
    data = get_breast_cancer_data()

    # Get the dependent and independent features
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Split the data to training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform standardization on split data. Standardization is performed on independent features
    # only i.e. X_train/X_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Get the machine learning model selected by user
    model = models[model_name]

    # Train the model
    model.fit(X_train, y_train)

    # Prediction for test data
    y_pred = model.predict(X_test)

    # Create data frame for storing model performance parameters
    performance_tests = ['Accuracy', 'F1 Score', 'Precision', 'Recall',
                         'ROC AUC Score']  # This will be the name of columns
    df_performance_metric = pd.DataFrame(columns=performance_tests)

    # Get the performance metrics and add them in the data frame
    df_performance_metric.loc[0, 'Accuracy'] = "{:.2f}".format(accuracy_score(y_test, y_pred))
    df_performance_metric.loc[0, 'F1 Score'] = "{:.2f}".format(f1_score(y_test, y_pred))
    df_performance_metric.loc[0, 'Precision'] = "{:.2f}".format(precision_score(y_test, y_pred))
    df_performance_metric.loc[0, 'Recall'] = "{:.2f}".format(recall_score(y_test, y_pred))
    df_performance_metric.loc[0, 'ROC AUC Score'] = "{:.2f}".format(roc_auc_score(y_test, y_pred))

    return model, scaler, df_performance_metric


# Function to predict outcome given the input data nad based on trained machine learning model
def model_predictions(input_data, model, scaler):
    # Convert the input data into a 2D array. This is required for machine learning model
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Scale the input data. This is required before making prediction
    input_array_scaled = scaler.transform(input_array)

    # Get prediction from the trained model
    prediction = model.predict(input_array_scaled)

    # Get prediction probability
    probability = model.predict_proba(input_array_scaled)

    return prediction, probability
