from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app = application

## import ridge regresson and standard scaler pickle files
ridge_model = pickle.load(open('models/ridge.pkl' , 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl' , 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Get all 8 features from form
            MedInc = float(request.form['MedInc'])
            HouseAge = float(request.form['HouseAge'])
            AveRooms = float(request.form['AveRooms'])
            AveBedrms = float(request.form['AveBedrms'])
            Population = float(request.form['Population'])
            AveOccup = float(request.form['AveOccup'])
            Latitude = float(request.form['Latitude'])
            Longitude = float(request.form['Longitude'])

            # Prepare input
            input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                                    Population, AveOccup, Latitude, Longitude]])
            scaled_input = standard_scaler.transform(input_data)
            predicted_value = ridge_model.predict(scaled_input)[0]

            return render_template('home.html', result=round(predicted_value, 2))
        except Exception as e:
            return f"Error occurred: {str(e)}"
    else:
        return render_template('home.html', result=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0')