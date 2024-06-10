from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo1.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request y convertirlos a flotantes
        genus = float(request.form['genus'])
        class_id = float(request.form['class_id'])
        snake_sub_family = float(request.form['snake_sub_family'])
        poisonous = float(request.form['poisonous'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[genus, class_id, snake_sub_family, poisonous]], columns=['genus', 'class_id', 'snake_sub_family', 'poisonous'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        predicted_class = str(prediction[0])  # Convertir la predicción a cadena
        
        app.logger.debug(f'Predicción: {predicted_class}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': predicted_class})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': 'Error en la solicitud. Detalles en el registro del servidor.'}), 400

if __name__ == '__main__':
    app.run(debug=True)

