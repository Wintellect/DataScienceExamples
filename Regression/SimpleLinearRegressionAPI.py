from flask import Flask, jsonify, request
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            years_of_experience = float(data["yearsOfExperience"])
        except ValueError:
            return jsonify("Please enter a number.")

        return jsonify(lin_reg.predict(years_of_experience).tolist())


@app.route("/retrain", methods=['POST'])
def retrain():
    if request.method == 'POST':
        data = request.get_json()

        training_set = joblib.load("./training_data.pkl")
        training_labels = joblib.load("./training_labels.pkl")

        df = pd.read_json(data)

        df_training_set = df["YearsExperience"]
        df_training_labels = df["Salary"]

        lin_reg = LinearRegression()
        lin_reg.fit(df_training_set, df_training_labels)


@app.route("/currentscore", methods=['GET'])
def current_score():
    if request.method == 'GET':
        pass


if __name__ == '__main__':
    lin_reg = joblib.load("./linear_regression_model.pkl")
    app.run(debug=True)
