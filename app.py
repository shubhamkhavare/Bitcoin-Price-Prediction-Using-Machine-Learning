from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_price():
    days = int(request.form.get('days'))

    # prediction
    #for i in range(days):
    return str(model.predict(np.array([days]).reshape(-1, 1)))

if __name__ == '__main__':
    app.run(debug=True)
