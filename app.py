from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Define mappings for categorical features
feature_mappings = {
    "Gender": {0: "Male", 1: "Female"},
    "Adenopathy": {0: "No", 1: "Yes"},
    "Pathology": {0: "Papillary", 1: "Micropapillary", 2: "Follicular", 3: "Hurthel cell"},
    "Focality": {0: "Unifocal", 1: "Multifocal"},
    "Risk": {0: "Low", 1: "High"},
    "T": {0: "T1", 1: "T2", 2: "T3", 3: "T4"},
    "N": {0: "N0", 1: "N1"},
    "M": {0: "M0", 1: "M1"},
    "Stage": {0: "I", 1: "II", 2: "III", 3: "IV"},
    "Response": {0: "Excellent", 1: "Structural Incomplete", 2: "Indeterminate", 3: "Biochemical Incomplete"},
    "Hx Radiothreapy": {0: "No", 1: "Yes"}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        features = [
            int(request.form.get('Age')),
            int(request.form.get('Gender')),
            int(request.form.get('Hx Radiothreapy')),
            int(request.form.get('Adenopathy')),
            int(request.form.get('Pathology')),
            int(request.form.get('Focality')),
            int(request.form.get('Risk')),
            int(request.form.get('T')),
            int(request.form.get('N')),
            int(request.form.get('M')),
            int(request.form.get('Stage')),
            int(request.form.get('Response'))
        ]
        # Make prediction
        prediction = model.predict([features])
        result = "Yes" if prediction[0] == 1 else "No"
        return render_template('result.html', result=result)
    return render_template('index.html', feature_mappings=feature_mappings)

if __name__ == "__main__":
    app.run(debug=True)