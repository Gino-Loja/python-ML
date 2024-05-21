import joblib  # importa las bibliotecas joblib para cargar el
# modelo y numpy y Flask para crear la aplicación web.
import numpy as np
from flask import Flask
from flask import jsonify
app = Flask(__name__)
# POSTMAN PARA PRUEBAS


@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([4.929435188, 4.728564805,1.054698706,1.384788632,0.18708007,0.479246736,0.13936238,0.072509497,1.510908604
                       ])
    prediction = model.predict(X_test.reshape(1, -1))
    return jsonify({'prediccion': list(prediction)})


if __name__ == "__main__":
    model = joblib.load('./models/best_model_0.932.pkl')
    app.run(port=8080)
