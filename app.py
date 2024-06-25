from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging 

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('electricidad.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        X1 = float(request.form['X1'])
        X2 = float(request.form['X2'])
        X4 = float(request.form['X4'])
        X7 = float(request.form['X7'])
        X8 = float(request.form['X8'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[X1, X2, X4, X7, X8]], columns=['X1', 'X2', 'X4', 'X7', 'X8'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
