from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('student_score_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours_studied = float(request.form['hours_studied'])
        sleep_hours = float(request.form['sleep_hours'])
        distraction_level = float(request.form['distraction_level'])
        attendance = float(request.form['attendance'])
        past_score = float(request.form['past_score'])  # ✅ Now used

        # ✅ Now 5 features
        features = np.array([[hours_studied, attendance, sleep_hours, distraction_level, past_score]])
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('result.html', prediction_text=f"🎯 Predicted Score: {output}%")

    except Exception as e:
        return f"❌ Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
    