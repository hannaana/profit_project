from flask import Flask, request, jsonify, render_template
import pickle
from predict import Predictor
import json


app = Flask(__name__)  # initialize the Flask App
model = pickle.load(open("finalized_model.pkl", 'rb'))
predictor = Predictor()

# @app.route("/",methods=['POST'])
# def predict():
#     data=request.get_json(force=True)
#     output = predictor.predict_for_new_sample(data)
#     return render_template('index.html', prediction_text='Profit should be $ {}'.format(output))

@app.route('/predict',methods=['POST'])
def results():
    #data = request.get_json(force=True)
    data = request.get_json(force=False)
    print(type(data))
    output = predictor.predict_for_new_sample(data)
    return jsonify(output)
    #return jsonify("yes")


if __name__ == "__main__":
    app.run(debug=True)


