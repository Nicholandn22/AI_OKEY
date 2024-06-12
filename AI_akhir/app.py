from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if not file:
            return render_template('index.html', error='No file uploaded')

        # Read the CSV file
        try:
            data = pd.read_csv(file)
        except Exception as e:
            return render_template('index.html', error=str(e))

        # Preprocess the data if necessary (e.g., normalization)
        data_scaled = scaler.transform(data)  # Adjust according to your preprocessing steps

        # Perform prediction
        try:
            predictions = model.predict(data_scaled)
            result = ['Clean' if pred == 3 else 'Not Clean' for pred in predictions]
            data['Prediction'] = result
            output_filename = 'output.csv'
            data.to_csv(output_filename, index=False)
        except Exception as e:
            return render_template('index.html', error=str(e))

        return render_template('index.html', prediction_table=data.to_html(), output_filename=output_filename)

if __name__ == '__main__':
    app.run(debug=True)
