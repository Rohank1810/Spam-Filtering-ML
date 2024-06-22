from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    with open(os.path.dirname(os.path.realpath(__file__)) + '/spam_classifier.pkl', 'rb') as model_file:
        cv, clf = pickle.load(model_file)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run()
