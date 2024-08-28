#Librerias
from flask import Flask, request, jsonfy, render_template
import numpy as np
import joblib
#Files management
import os
from werkzeug.utils import secure_filename

#Load model
dt=joblib.load('dt.joblib')
#Create Flask app
server = Flask(__name__)

#Define a route to send JSON -data
@server.route('/predictjson', methods=['POST'])
def predictjson():
    #Procesar datos de entrada
    data = request.json
    print(data)
    inputData = np.array([
        data['pH'],
        data['sulphates'],
        data['alcohol']
    ])
    #predecir utilizando la entrada y el modelo
    result = dt.predict(inputData.reshape(1,-1))
    #Enviar respuesta
    return jsonfy({'Prediction': str(result[0])})
if __name__ == '__main__':
    server.run(debug=False, host='0.0.0.0', port=8080)