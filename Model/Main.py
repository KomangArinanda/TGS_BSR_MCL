from flask import Flask, jsonify, request
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
import pandas as pandas


app = Flask(__name__)
Swagger(app)
# CORS(app)

@app.route('/predict/task', methods=['POST'])
def predict():
    """
    Ini Adalah Endpoint Untuk Memprediksi IRIS
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: Petal
          required:
            - petalLength
            - petalWidth
            - sepalLength
            - sepalWidth
          properties:
            petalLength:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
            petalWidth:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
            sepalLength:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
            sepalWidth:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    milk = new_task['milk']
    toothed = new_task['toothed']
    eggs = new_task['eggs']
    feathers = new_task['feathers']
    backbone = new_task['backbone']
    legs = new_task['legs']
    breathes = new_task['breathes']
    tail = new_task['tail']
    fins = new_task['fins']
    aquatic = new_task['aquatic']

    X_New = np.array([[legs, toothed, backbone, tail, breathes, feathers, milk,aquatic, fins, eggs]])
    # X_New = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 4, 0, 0, 0]])
    clf = joblib.load('randomForestClassifier.pkl')

    resultPredict = clf[0].predict(X_New)

    return jsonify({'message': format(clf[1][resultPredict-1])})


if __name__ == '__main__':
    app.run()