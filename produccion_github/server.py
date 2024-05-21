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
        2923, 549902, 34023, 13142, 54, 24.1, 78366,
        63868, 10.4, 1.7, 3224, 29, 17258, 556,
        168, 783618, 259030, 600, 3411, 5731,
        7195, 321, 190166, 25.3, 59925, 20.5,
        181146, 1, 5, 17258, 0, 0, 1.1, 0.987808835,
        16514, 622, 0, 7065, 5, 2304, 5, 77996, 215730, 107, 5
    ])
    prediction = model.predict(X_test.reshape(1, -1))

    print(list(prediction)[0])
    return jsonify({'prediccion': str(prediction[0])})


if __name__ == "__main__":
    model = joblib.load('./models/best_model_0.965.pkl')
    app.run(port=8081)
