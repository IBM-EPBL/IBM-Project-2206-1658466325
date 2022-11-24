import numpy as np
from flask import Flask,render_template,request
import pickle
app = Flask(__name__)
model = pickle.load(open('decmodel.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]  
    prediction = model.predict(features) 
    output = prediction
    if output == 1:
        return render_template('index.html',result="Kindly check with your cardiologist..this test says you are suffering from heart disease...")
    else:
        return render_template('index.html',result="Hurray !! The Patient is Healthy and not likely to have Heart Disease if the health is maintained...")
    
if __name__ == "__main__":
    app.run(debug=True)
