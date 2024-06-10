from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging 

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modeloDiabeteSEscalar.pyl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        glucosa = float(request.form['glucosa'])
        presion_sanguinea = float(request.form['presionSanguinea'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetesPedigreeFunction'])
        edad = float(request.form['edad'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[glucosa, presion_sanguinea, bmi, diabetes_pedigree_function, edad]], columns=['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'diabetes': "si tiene diabetes" if prediction[0] == 1 else "no tiene diabetes"})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
