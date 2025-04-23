from flask import Flask, render_template, request, jsonify
import sys
import os
import csv
import random

# Agrega el backend al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from inference import SentimentInference

app = Flask(__name__)

# Cargar modelo
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
    examples = {"positivo": [], "negativo": []}
    try:
        with open(dataset_path, 'r', encoding='latin-1') as file:
            reader = csv.reader(file)
            positive_tweets = []
            negative_tweets = []
            count = 0
            for row in reader:
                try:
                    sentiment = int(row[0].strip('"'))
                    tweet_text = row[5]
                    if sentiment == 0 and 20 <= len(tweet_text) <= 140 and len(negative_tweets) < 50:
                        negative_tweets.append(tweet_text)
                    if count > 800000 and sentiment == 4 and 20 <= len(tweet_text) <= 140:
                        positive_tweets.append(tweet_text)
                        if len(positive_tweets) >= 50:
                            break
                    count += 1
                    if len(negative_tweets) >= 50 and count < 800000:
                        for _ in range(800000 - count):
                            next(reader)
                        count = 800000
                except:
                    pass
            if positive_tweets:
                examples["positivo"] = random.sample(positive_tweets, min(num_examples, len(positive_tweets)))
            if negative_tweets:
                examples["negativo"] = random.sample(negative_tweets, min(num_examples, len(negative_tweets)))
        return examples
    except:
        return examples

@app.route('/')
def index():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'training.1600000.processed.noemoticon.csv'))
    examples = get_examples_from_dataset(dataset_path)
    return render_template('index.html', model_status=model_loaded, examples=examples)

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model_loaded:
        return jsonify({'error': 'Modelo no cargado.'}), 500

    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({'error': 'Texto inválido'}), 400

    try:
        result = inference_engine.predict(data['text'])
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
    except Exception as e:
        return jsonify({'error': f'Error en predicción: {str(e)}'}), 500

@app.route('/status')
def status():
    return jsonify({'model_loaded': model_loaded})

if __name__ == '__main__':
    app.run(debug=True)
