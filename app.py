from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('LSTM(2).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data from HTML
    param1 = float(request.form['param1'])  # Temperature
    param2 = float(request.form['param2'])  # Fuel Price
    param3 = float(request.form['param3'])  # CPI
    param4 = float(request.form['param4'])  # Unemployment
    param5 = int(request.form['param5'])    # Year
    param6 = int(request.form['param6'])    # Month
    
    # Create a feature array for prediction
    features = np.array([[param1, param2, param3, param4, param5, param6]])
    prediction = model.predict(features)
    
    return render_template('index.html', prediction=f'The predicted sales are: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)