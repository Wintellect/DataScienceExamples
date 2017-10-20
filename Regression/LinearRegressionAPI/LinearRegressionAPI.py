from flask import Flask, jsonify, request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            years_of_experience = float(request.form["yearsOfExperience"])
        except ValueError:
            return jsonify("Please enter a number.")

        return jsonify(lin_reg.predict(years_of_experience).tolist())


if __name__ == '__main__':
    lin_reg = joblib.load("../linear_regression_model.pkl")
    app.run(debug=True)