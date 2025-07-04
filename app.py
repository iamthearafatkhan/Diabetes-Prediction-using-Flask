from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = tf.keras.models.load_model('diabetes_ann_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Print form data for debugging
        print(request.form)

        # Get form data
        age = float(request.form['age'])
        gender = 1 if request.form['gender'].lower() == 'male' else 0
        pulse_rate = float(request.form['pulse_rate'])
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        family_diabetes = int(request.form['family_diabetes'])
        hypertensive = int(request.form['hypertensive'])

        # Prepare input
        input_data = np.array([[age, gender, pulse_rate, glucose, bmi, family_diabetes, hypertensive]])
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)
        probability = prediction[0][0]
        result = "Diabetic" if probability > 0.5 else "Not Diabetic"

        return render_template('result.html', result=result, probability=round(probability, 4))
    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
