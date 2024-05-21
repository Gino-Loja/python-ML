import joblib  # importa las bibliotecas joblib para cargar el
# modelo y numpy y Flask para crear la aplicación web.
import numpy as np
from flask import Flask
from flask import jsonify
app = Flask(__name__)
# POSTMAN PARA PRUEBAS


@app.route('/', methods=['GET'])
def predict():
    X_test = np.array([
     1,0,0.007837302,26.29881944,75.07375645,20.78440079,0.087797619,0.743551587,163.4945437])
    prediction = model.predict(X_test.reshape(1, -1))

    print(list(prediction)[0])
    return jsonify({'prediccion': str(prediction[0])})


if __name__ == "__main__":
    model = joblib.load('./out/best_model_1.0.pkl')
    app.run(port=8080)
