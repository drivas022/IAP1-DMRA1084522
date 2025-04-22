import pickle

class SentimentInference:
    def __init__(self, model_path='models/model.pkl', preprocessor_path='models/preprocessor.pkl'):
        """
        Inicializa el motor de inferencia cargando el modelo y el preprocesador
        
        model_path: ruta al archivo del modelo guardado
        preprocessor_path: ruta al archivo del preprocesador guardado
        """
        # Cargar el modelo
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Cargar el preprocesador
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
            
        # Mapeo de etiquetas numéricas a texto
        self.label_map = {
            0: "negativo",
            1: "positivo"
        }
        
        # Inicializar con algunos ejemplos para verificar que el modelo funcione
        self.examples = {
            "positivo": [
                "I love this new feature! It's amazing!",
                "Just had the best day ever at the beach!",
                "This movie was fantastic, highly recommend it"
            ],
            "negativo": [
                "Terrible service at the restaurant today",
                "I hate when my phone battery dies so quickly",
                "This app keeps crashing and it's frustrating"
            ]
        }
    
    def predict(self, text):
        """
        Predice el sentimiento de un texto
        
        text: texto a analizar
        
        Retorna: Un diccionario con la etiqueta predicha y las probabilidades
        """
        # Preprocesar el texto
        tokens = self.preprocessor.preprocess(text)
        
        # Realizar la predicción
        probas = self.model.predict_proba([tokens])[0]
        prediction = self.model.predict([tokens])[0]
        
        # Formatear resultados
        result = {
            "prediction": self.label_map.get(prediction, "desconocido"),
            "probabilities": {
                "negativo": probas.get(0, 0.0),
                "positivo": probas.get(1, 0.0)
            },
            "confidence": max(probas.values())
        }
        
        return result
    
    def analyze_batch(self, texts):
        """
        Analiza un lote de textos
        
        texts: lista de textos a analizar
        
        Retorna: Una lista de resultados
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
            
        return results

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del motor de inferencia
    inference_engine = SentimentInference()
    
    # Ejemplos de tweets
    test_tweets = [
        "I love this new phone! It's amazing and works perfectly.",
        "This movie was terrible, waste of time and money.",
        "The weather is nice today."
    ]
    
    # Analizar cada tweet
    for tweet in test_tweets:
        result = inference_engine.predict(tweet)
        
        print(f"\nTexto: {tweet}")
        print(f"Predicción: {result['prediction']} (confianza: {result['confidence']:.4f})")
        print(f"Probabilidades: {result['probabilities']}")