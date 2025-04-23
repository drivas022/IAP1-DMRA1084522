from flask import Flask, render_template, request, jsonify
import sys
import os
import csv
import random

# Añadir el directorio parent al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar el motor de inferencia
from backend.inference import SentimentInference

app = Flask(__name__)

# Inicializar el motor de inferencia
try:
    inference_engine = SentimentInference(
        model_path='../backend/models/model.pkl',
        preprocessor_path='../backend/models/preprocessor.pkl'
    )
    model_loaded = True
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model_loaded = False

def get_examples_from_dataset(dataset_path, num_examples=3):
    """
    Obtiene ejemplos reales del dataset para mostrar en la interfaz
    
    dataset_path: ruta al archivo CSV del dataset
    num_examples: número de ejemplos a extraer para cada clase
    
    Retorna: diccionario con ejemplos positivos y negativos
    """
    examples = {
        "positivo": [],
        "negativo": []
    }
    
    try:
        if not os.path.exists(dataset_path):
            print(f"No se encontró el archivo del dataset en {dataset_path}")
            return examples
            
        with open(dataset_path, 'r', encoding='latin-1') as file:
            reader = csv.reader(file)
            positive_tweets = []
            negative_tweets = []
            
            # Para el dataset Sentiment140:
            # - Los primeros 800,000 tweets son negativos (etiqueta 0)
            # - Los últimos 800,000 tweets son positivos (etiqueta 4)
            
            print("Buscando ejemplos de tweets...")
            
            count = 0
            for row in reader:
                try:
                    sentiment = int(row[0].strip('"'))
                    tweet_text = row[5]
                    
                    # Recolectar algunos tweets negativos del inicio
                    if sentiment == 0 and 20 <= len(tweet_text) <= 140 and len(negative_tweets) < 50:
                        negative_tweets.append(tweet_text)
                    
                    # Continuar hasta encontrar los tweets positivos (después de la fila 800,000)
                    if count > 800000 and sentiment == 4 and 20 <= len(tweet_text) <= 140:
                        positive_tweets.append(tweet_text)
                        if len(positive_tweets) >= 50:  # Suficientes ejemplos positivos
                            break
                    
                    count += 1
                    
                    # Mostrar progreso
                    if count % 100000 == 0:
                        print(f"Procesados {count} registros... ({len(positive_tweets)} positivos encontrados)")
                    
                    # Si ya tenemos suficientes ejemplos negativos y estamos cerca de los positivos
                    if len(negative_tweets) >= 50 and count < 800000:
                        # Saltar directamente a donde deberían estar los positivos
                        for _ in range(800000 - count):
                            next(reader)
                        count = 800000
                        print("Saltando a la sección de tweets positivos...")
                        
                except Exception as e:
                    if count < 10:
                        print(f"Error en fila {count}: {e}")
                    pass
            
            # Seleccionar aleatoriamente los ejemplos
            if positive_tweets:
                examples["positivo"] = random.sample(positive_tweets, min(num_examples, len(positive_tweets)))
            if negative_tweets:
                examples["negativo"] = random.sample(negative_tweets, min(num_examples, len(negative_tweets)))
            
            print(f"Debug: Encontrados {len(positive_tweets)} tweets positivos y {len(negative_tweets)} tweets negativos")
                
        return examples
        
    except Exception as e:
        print(f"Error al leer ejemplos del dataset: {e}")
        return examples

@app.route('/')
def index():
    """
    Ruta principal que muestra la página de inicio
    """
    # Ruta al dataset
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'training.1600000.processed.noemoticon.csv'))
    
    # Obtener ejemplos reales del dataset
    examples = get_examples_from_dataset(dataset_path)
    
    # Si no se pudieron obtener ejemplos del dataset, usamos valores por defecto
    if not examples["positivo"] and not examples["negativo"]:
        print("No se pudieron cargar ejemplos del dataset, usando ejemplos por defecto")
        examples = {
            "positivo": [],
            "negativo": []
        }
    
    return render_template('index.html', model_status=model_loaded, examples=examples)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Ruta para analizar un texto recibido por POST
    """
    if not model_loaded:
        return jsonify({
            'error': 'El modelo no está cargado correctamente. Por favor, entrena el modelo primero.'
        }), 500
    
    # Obtener el texto del request
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'No se proporcionó ningún texto para analizar.'
        }), 400
    
    text = data['text']
    
    # Validar que el texto no esté vacío
    if not text.strip():
        return jsonify({
            'error': 'El texto está vacío. Por favor, ingresa un texto para analizar.'
        }), 400
    
    try:
        # Realizar la predicción
        result = inference_engine.predict(text)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
    except Exception as e:
        return jsonify({
            'error': f'Error al analizar el texto: {str(e)}'
        }), 500

@app.route('/status')
def status():
    """
    Ruta para verificar el estado del modelo
    """
    return jsonify({
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    app.run(debug=True)