import numpy as np
import pandas as pd
import pickle
import os
from flask import Flask, render_template, request

app = Flask(__name__)

import os
print(os.getcwd())

# Load Model and Scaler locally from the same folder
with open("Training/rdf.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("Training/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

print("Loaded Model:", model)
print("Loaded Model Type:", type(model))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("input.html")

@app.route("/submit", methods=["POST"])
def submit():
    try:
        # Read Form Values
        Gender = float(request.form["Gender"])
        Married = float(request.form["Married"])
        Dependents = float(request.form["Dependents"])
        Education = float(request.form["Education"])
        Self_Employed = float(request.form["Self_Employed"])
        ApplicantIncome = float(request.form["ApplicantIncome"])
        CoapplicantIncome = float(request.form["CoapplicantIncome"])
        LoanAmount = float(request.form["LoanAmount"]) / 1000.0
        Loan_Amount_Term = float(request.form["Loan_Amount_Term"])
        Credit_History = float(request.form["Credit_History"])
        Property_Area = float(request.form["Property_Area"])

        # Define the exact feature names in order
        feature_names = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
            "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]

        # Original Features List
        features = [[
            Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Property_Area
        ]]

        print("\n============================")
        print("Original Features")
        print(features)

        # Convert to DataFrame FIRST to avoid the scikit-learn UserWarning
        features_df = pd.DataFrame(features, columns=feature_names)

        # Scale using the DataFrame with valid feature names
        scaled = scaler.transform(features_df)

        print("\nScaled Features")
        print(scaled)

        # Create the structured DataFrame for prediction
        data = pd.DataFrame(scaled, columns=feature_names)

        print("\nDataFrame")
        print(data)

        print(model)
        print(type(model))

        # Prediction
        prediction = model.predict(data)

        print("\nPrediction = ", prediction)

        if prediction[0] == 1:
            result = "✅ Loan Approved"
        else:
            result = "❌ Loan Rejected"

        return render_template("output.html", result=result)

    except Exception as e:
        return f"<h2>Error:</h2><pre>{e}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)