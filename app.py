import os
import pandas as pd
from flask import Flask, jsonify, request

from model import train_model
from config import settings

app = Flask(__name__)

model, vectorizer_obj = train_model(path=os.path.abspath("heart.csv"))


@app.route('/')
@app.route('/index')
def index():
    return '''
<html>
    <head>
        <title>Heart diseases</title>
    </head>
    <body>
        <h1>This is app to classify heart diseases</h1>
    </body>
</html>'''


@app.route("/predict", methods=['POST'])
def make_prediction():
    posted_data = request.get_json()
    df = pd.DataFrame.from_dict([posted_data], orient='columns')
    vectorized_df = vectorizer_obj.vectorize(df)
    predictions = model.predict(vectorized_df.values)
    res = [settings.TARGET_EXPLANATION[i] for i in predictions]
    return jsonify({'status': res[0]})


@app.route('/model')
def get_model():
    return jsonify({'name': 'Logistic regression',
                    'accuracy': model.accuracy})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)



