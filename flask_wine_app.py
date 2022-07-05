import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('toy_data_wineclassification.pkl', 'rb'))
print(model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)

    output = prediction
    #print(output)
    return render_template('index.html', prediction_text='Classified wine is of class {}'.format(output))

if __name__ == '__main__':
    app.run(port=5000, debug=True)