from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("modello_ritardo_arrivo.pkl")

@app.route("/")
def home():
    return "🛬 ML API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # features = [compagnia_enc, origine_enc, destinazione_enc, giorno, ora]
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"ritardo_minuti": round(prediction, 2)})
