import pickle
from flask import Flask, request, jsonify, render_template
from ml_model import predict_review

app = Flask(__name__)
model = pickle.load(open('ml_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    comment = str(request.form['review'])
    # print(comment)
    predictions = predict_review(comment, model)
    res = predictions[0]
    return render_template('index.html', prediction_text='This is a {} review!!'.format(res))

# @app.route('/', methods=['GET'])
# def ping():
#     return "Pinging Model!!"


if __name__ == '__main__':
    app.run(debug=True)
