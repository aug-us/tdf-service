from __future__ import division, print_function
# coding=utf-8
import sys
sys.path.append('./path/src/')
import pickle
import base64

from tensorflow import keras
from datetime import datetime, date
from dateutil import relativedelta
import matplotlib.pyplot as plt



from Eval import inverscale

# Flask utils
from flask import Flask, request, render_template

# Define a flask app
app = Flask(__name__)


with open('./path/models/reframed.pkl','rb') as a:
    data = pickle.load(a)
    values = data.values
with open('./path/models/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
model = keras.models.load_model('./path/models/model.h5')
print(model)
  

@app.route('/')
def index():
    # Main page
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        #getting all the variables from the form
        start_date = request.form['start_date']
        end_date = request.form['end_date']
    
    print(start_date, end_date)
    
    start_datetime = datetime.strptime(start_date, '%Y-%m').date()
    print(start_datetime)
    
    end_datetime = datetime.strptime(end_date, '%Y-%m').date()
    print(end_datetime)
    
    ref_date = date(2012, 1, 1)
    print(ref_date)
    
    start_datetime_diff = relativedelta.relativedelta(start_datetime,ref_date)
    start_datetime_diff = start_datetime_diff.months + start_datetime_diff.years * 12
    print(start_datetime_diff)
    
    end_datetime_diff = relativedelta.relativedelta(end_datetime,ref_date)
    end_datetime_diff = end_datetime_diff.months + end_datetime_diff.years * 12
    print(end_datetime_diff)
    
    #train = values[:start_datetime_diff-1,:]
    test = values[start_datetime_diff-1:end_datetime_diff-1,:]
    test_X, test_Y = test[:,:-1],test[:,-1]
    test_X   = test_X.reshape(test_X.shape[0],12,int(test_X.shape[1]/12))
    print(test_X.shape)
    
    predictions = list()
    groundtrue = list()

    for i in range(len(test_X)):
        r,f,c = test_X.shape
        test = test_X[i].reshape(1,f,c)
        yhat = model.predict(test)
        inv_yhat,inv_y = inverscale(yhat,test_X[i],test_Y[i],scaler)
        predictions.append(inv_yhat)
        groundtrue.append(inv_y)
    
    print(predictions)
    volumes_pred = [list(x) for x in predictions]
    print(volumes_pred)
    volumes_true = [list(x) for x in groundtrue]
    print(volumes_true)
    
    plt.plot(volumes_pred)
    plt.plot(volumes_true)
    plt.xlabel('Months', fontsize=10)
    plt.ylabel('Volume of arrival', fontsize=10)
    plt.legend(["Predictions", "True value"], loc ="upper right")
    plt.savefig('static/plot.jpg')
    plt.show()
    
    data_uri = base64.b64encode(open('static/plot.jpg', 'rb').read()).decode('utf-8')
    img_tag = "data:image/png;base64,{0}".format(data_uri)
    print(img_tag)
    # Make prediction
    return render_template('output.html', img = img_tag)


if __name__ == '__main__':
    
    app.run()