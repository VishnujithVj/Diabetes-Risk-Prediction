# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest Classifier model
filename = 'C:/Users/LENOVO/rf_diabetes_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        gen = request.form['gender']
        pl = float(request.form['polyuria'])
        pd = float(request.form['polydipsia'])
        swl = float(request.form['sudden_weight_loss'])
        wk = float(request.form['weakness'])
        pg = float(request.form['polyphagia'])
        gt = float(request.form['genital_thrush'])
        vb = float(request.form['visual_blurring'])
        ich = float(request.form['itching'])
        irtb = float(request.form['irritability'])
        dh = float(request.form['delayed_healing'])
        pp = float(request.form['partial_paresis'])
        ms = float(request.form['muscle_stiffness'])
        ap = float(request.form['alopecia'])
        obs = float(request.form['obesity'])
        
        # Preprocess the gender input
        gen_encoded = 0 if gen == 'male' else 1
        
        data = np.array([[age, gen_encoded, pl, pd, swl, wk, pg, gt, vb, ich, irtb, dh, pp, ms, ap, obs]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
