from flask import Flask, request, jsonify
import pickle
from predict import Predictor
import json
import sqlite3


app = Flask(__name__)  # initialize the Flask App
model = pickle.load(open("finalized_model.pkl", 'rb'))
predictor = Predictor()

@app.route("/predict", methods=['POST'])
def index():
    data = request.get_json(force=True)
    print(data)
    output = predictor.predict_for_new_sample(data)

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)


