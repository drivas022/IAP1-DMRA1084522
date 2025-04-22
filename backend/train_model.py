import os
import csv
import pickle
import numpy as np
from preprocessor import Preprocessor
from naive_bayes import NaiveBayes
import time
import gc  # Garbage Collector para liberar memoria

def load_sentiment140_dataset(file_path, limit=None):
    """
    Carga el dataset Sentiment140
    
    file_path: ruta al archivo CSV
    limit: número máximo de filas a cargar (opcional, para pruebas)
    
    Retorna: X (textos), y (etiquetas)
    """
    X = []
    y = []
    
    print(f"Cargando dataset desde {file_path}...")
    
    # Cargamos el dataset con la codificación correcta
    with open(file_path, 'r', encoding='latin-1') as file:
        csv_reader = csv.reader(file)
        count = 0
        pos_count = 0
        neg_count = 0
        
        for row in csv_reader:
            # La estructura del CSV de Sentiment140 es:
            # target (0 = negativo, 4 = positivo), id, fecha, query, usuario, texto
            try:
                sentiment = int(row[0].strip('"'))  # Limpiamos posibles comillas
                text = row[5]
                
                # Mapear 0->0 (negativo), 4->1 (positivo)
                if sentiment == 0:
                    label = 0  # Negativo
                    neg_count += 1
                elif sentiment == 4:
                    label = 1  # Positivo
                    pos_count += 1
                else:
                    continue  # Ignorar otros valores
                    
                X.append(text)
                y.append(label)
                
                count += 1
                if limit and count >= limit:
                    break
                
                # Mostrar progreso más frecuentemente para un proceso largo
                if count % 100000 == 0:
                    print(f"Procesados {count} tweets ({pos_count} positivos, {neg_count} negativos)")
                
            except (ValueError, IndexError) as e:
                print(f"Error procesando fila: {e}")
                continue
                
    print(f"Dataset cargado: {len(X)} ejemplos ({pos_count} positivos, {neg_count} negativos)")
    return X, np.array(y)

def main():
    """
    Función principal para entrenar el modelo
    """
    # Ruta al dataset
    dataset_path = 'dataset/training.1600000.processed.noemoticon.csv'
    
    # Verificar si el directorio de modelo existe, si no, crearlo
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Cargar dataset completo (sin límite)
    # Esto cargará los 1.6 millones de tweets
    start_time_total = time.time()
    X, y = load_sentiment140_dataset(dataset_path)
    
    # Preprocesamiento
    print("Preprocesando textos...")
    preprocessor = Preprocessor()
    X_preprocessed = []
    
    start_time = time.time()
    batch_size = 50000  # Procesar en lotes para liberar memoria
    for i, text in enumerate(X):
        tokens = preprocessor.preprocess(text)
        X_preprocessed.append(tokens)
        
        # Mostrar progreso (cada 50,000 tweets)
        if (i+1) % 50000 == 0:
            print(f"Preprocesados {i+1}/{len(X)} textos ({(i+1)/len(X)*100:.1f}%)")
            # Llamar al garbage collector para liberar memoria
            gc.collect()
    
    preprocess_time = time.time() - start_time
    print(f"Preprocesamiento completado en {preprocess_time:.2f} segundos")
    
    # Liberar memoria eliminando la lista original de textos
    del X
    gc.collect()
    
    # Dividir en conjuntos de entrenamiento y prueba (80% / 20%)
    split_idx = int(len(X_preprocessed) * 0.8)
    X_train, X_test = X_preprocessed[:split_idx], X_preprocessed[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Conjunto de entrenamiento: {len(X_train)} ejemplos")
    print(f"Conjunto de prueba: {len(X_test)} ejemplos")
    
    # Liberar más memoria
    del X_preprocessed
    gc.collect()
    
    # Entrenar modelo
    print("Entrenando modelo Naive Bayes...")
    print("Este proceso puede tomar varias horas con el dataset completo.")
    print("Por favor, sea paciente y no interrumpa el proceso.")
    start_time = time.time()
    
    model = NaiveBayes()
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Entrenamiento completado en {train_time:.2f} segundos")
    
    # Evaluar modelo
    print("Evaluando modelo...")
    metrics, confusion_matrix = model.get_metrics(X_test, y_test)
    
    print("\nMétricas de evaluación:")
    for class_label, class_metrics in metrics.items():
        if class_label == 'avg':
            print(f"\nPromedios:")
        else:
            print(f"\nClase {class_label}:")
        for metric_name, metric_value in class_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\nMatriz de confusión:")
    print(confusion_matrix)
    
    # Guardar modelo y preprocessor
    print("Guardando modelo...")
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print("Modelo guardado en 'models/model.pkl'")
    print("Preprocesador guardado en 'models/preprocessor.pkl'")
    
    # Tiempo total
    total_time = time.time() - start_time_total
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Tiempo total de ejecución: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()