from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'age': int(request.form['age']),
            'amount': int(request.form['amount']),
            'duration': int(request.form['duration']),
            'status': request.form['status'],
            'credit_history': request.form['credit_history'],
            'purpose': request.form['purpose'],
            'savings': request.form['savings'],
            'employment_duration': request.form['employment'],
            'personal_status_sex': request.form['personal_status'],
            'other_debtors': request.form['other_debtors'],
            'present_residence': int(request.form['residence']),
            'property': request.form['property'],
            'other_installment_plans': request.form['other_plans'],
            'housing': request.form['housing'],
            'job': request.form['job'],
            'telephone': request.form['telephone'],
            'foreign_worker': request.form['foreign_worker'],
            'installment_rate': int(request.form['installment_rate']),
            'number_credits': int(request.form['credits']),
            'people_liable': int(request.form['liable'])
        }

        df = pd.DataFrame([data])

        pred = model.predict(df)[0]

        if pred == 1:
            result = " High Risk Customer"
        else:
            result = " Low Risk Customer"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=str(e))


if __name__ == "__main__":
    app.run(debug=True)