from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models, scalers, and SMOTE
stacking_model = joblib.load('models/stacking_model.pkl')
scaler = joblib.load('models/scaler.pkl')
smote = joblib.load('models/smote.pkl')

# Load the data to calculate median values
data = pd.read_csv("C:\\Users\\angie\\OneDrive\\Desktop\\Final Project\\Final_Project_GHRacingPredictor_App\\data\\data_final.csv")

# Calculate the median values for each column
median_values = data.median()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Trap': [],
        'BSP': [],
        'Odds': [],
        'Public_Estimate': [],
        'Finish_All': [],
        'Races_All': [],
        'implied_probability': [],
        'win_percentage': [],
        'BSP_Odds_PublicEstimate': []
    }

    # Loop through each greyhound and calculate the required features
    for i in range(1, 7):
        Trap = float(request.form[f'Trap{i}'])
        BSP = float(request.form[f'BSP{i}'])
        Odds = float(request.form[f'Odds{i}'])
        Public_Estimate = float(request.form[f'Public_Estimate{i}'])
        Finish_All = float(request.form[f'Finish_All{i}'])
        Races_All = float(request.form[f'Races_All{i}'])

        # Calculate implied_probability
        implied_probability = 1 / Odds if Odds > 0 else 0

        # Calculate win_percentage
        win_percentage = Finish_All / Races_All if Races_All > 0 else 0

        # Calculate BSP_Odds_PublicEstimate
        BSP_Odds_PublicEstimate = BSP + Odds / Public_Estimate if Public_Estimate > 0 else BSP

        # Append the values to input_data
        input_data['Trap'].append(Trap)
        input_data['BSP'].append(BSP)
        input_data['Odds'].append(Odds)
        input_data['Public_Estimate'].append(Public_Estimate)
        input_data['Finish_All'].append(Finish_All)
        input_data['Races_All'].append(Races_All)
        input_data['implied_probability'].append(implied_probability)
        input_data['win_percentage'].append(win_percentage)
        input_data['BSP_Odds_PublicEstimate'].append(BSP_Odds_PublicEstimate)

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame(input_data)

    # Remove the 'Winner' column if it exists
    if 'Winner' in input_df.columns:
        input_df = input_df.drop(columns=['Winner'])

    # Ensure the column order matches the training data
    input_df = input_df[median_values.index]

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make predictions
    predictions = stacking_model.predict(input_scaled)
    prediction_proba = stacking_model.predict_proba(input_scaled)

    # Create a response
    response = {
        'predictions': predictions.tolist(),
        'prediction_probas': prediction_proba.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)