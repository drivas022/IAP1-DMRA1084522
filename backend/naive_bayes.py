import numpy as np
import math
from collections import defaultdict, Counter

class NaiveBayes:
    def __init__(self):
        # Probabilidades previas de las clases P(y)
        self.class_priors = {}
        
        # Probabilidades condicionales P(xi|y)
        self.feature_probs = {}
        
        # Vocabulario
        self.vocabulary = set()
        
        # Número de clases
        self.classes = []
        
        # Conteo de palabras para cada clase
        self.word_counts = {}
        
        # Conteo total de palabras para cada clase
        self.total_word_counts = {}
        
        # Parámetro para suavizado Laplace
        self.alpha = 1.0

    def fit(self, X, y):
        """
        Entrena el modelo Naive Bayes
        
        X: lista de listas, donde cada lista contiene las palabras de un documento
        y: lista de etiquetas de clase
        """
        # Obtener clases únicas
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Crear vocabulario
        for doc in X:
            self.vocabulary.update(doc)
        
        vocab_size = len(self.vocabulary)
        
        # Calcular probabilidades previas P(y)
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
            
        # Inicializar contadores
        self.word_counts = {c: defaultdict(int) for c in self.classes}
        self.total_word_counts = {c: 0 for c in self.classes}
        
        # Contar ocurrencias de palabras en cada clase
        for doc, label in zip(X, y):
            for word in doc:
                self.word_counts[label][word] += 1
                self.total_word_counts[label] += 1
                
        # Calcular probabilidades condicionales con suavizado Laplace
        self.feature_probs = {c: {} for c in self.classes}
        
        for c in self.classes:
            for word in self.vocabulary:
                # P(word|class) = (count(word, class) + alpha) / (total_words_in_class + alpha * vocab_size)
                numerator = self.word_counts[c][word] + self.alpha
                denominator = self.total_word_counts[c] + self.alpha * vocab_size
                self.feature_probs[c][word] = numerator / denominator
                
    def predict_proba(self, X):
        """
        Predice la probabilidad de cada clase para los documentos dados
        
        X: lista de listas, donde cada lista contiene las palabras de un documento
        """
        result = []
        
        for doc in X:
            class_scores = {}
            
            for c in self.classes:
                # Inicializar con el logaritmo de la probabilidad previa P(y)
                score = math.log(self.class_priors[c])
                
                # Sumar los logaritmos de las probabilidades condicionales P(xi|y)
                for word in doc:
                    if word in self.vocabulary:
                        score += math.log(self.feature_probs[c][word])
                    else:
                        # Palabra nueva, usar suavizado Laplace
                        score += math.log(self.alpha / (self.total_word_counts[c] + self.alpha * len(self.vocabulary)))
                
                class_scores[c] = score
                
            # Convertir a probabilidades (normalizar)
            result.append(self._normalize_log_probs(class_scores))
            
        return result
    
    def _normalize_log_probs(self, log_probs):
        """
        Normaliza los logaritmos de probabilidades a probabilidades
        """
        # Encontrar el valor máximo para estabilidad numérica
        max_log_prob = max(log_probs.values())
        
        # Restar el máximo (para evitar desbordamiento)
        exp_probs = {c: math.exp(log_prob - max_log_prob) for c, log_prob in log_probs.items()}
        
        # Normalizar
        total = sum(exp_probs.values())
        normalized = {c: p / total for c, p in exp_probs.items()}
        
        return normalized
    
    def predict(self, X):
        """
        Predice la clase para los documentos dados
        
        X: lista de listas, donde cada lista contiene las palabras de un documento
        """
        all_probs = self.predict_proba(X)
        predictions = []
        
        for probs in all_probs:
            # Seleccionar la clase con mayor probabilidad
            predictions.append(max(probs, key=probs.get))
            
        return predictions
    
    def score(self, X, y):
        """
        Calcula la precisión del modelo
        
        X: lista de listas, donde cada lista contiene las palabras de un documento
        y: lista de etiquetas de clase
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_metrics(self, X, y):
        """
        Calcula las métricas de evaluación del modelo (precision, recall, f1-score)
        
        X: lista de listas, donde cada lista contiene las palabras de un documento
        y: lista de etiquetas de clase
        """
        y_pred = self.predict(X)
        
        # Inicializar métricas por clase
        metrics = {}
        
        for c in self.classes:
            # Verdaderos positivos, falsos positivos, falsos negativos
            tp = np.sum((y == c) & (y_pred == c))
            fp = np.sum((y != c) & (y_pred == c))
            fn = np.sum((y == c) & (y_pred != c))
            
            # Calcular precision, recall, f1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[c] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1
            }
            
        # Calcular métricas promedio
        avg_precision = np.mean([m['precision'] for m in metrics.values()])
        avg_recall = np.mean([m['recall'] for m in metrics.values()])
        avg_f1 = np.mean([m['f1-score'] for m in metrics.values()])
        
        metrics['avg'] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1-score': avg_f1
        }
        
        # Calcular matriz de confusión
        cm = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        for i, true_class in enumerate(self.classes):
            for j, pred_class in enumerate(self.classes):
                cm[i, j] = np.sum((y == true_class) & (y_pred == pred_class))
                
        return metrics, cm