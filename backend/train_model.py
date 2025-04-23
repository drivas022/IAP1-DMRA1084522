import os
import csv
import pickle
import numpy as np
from preprocessor import Preprocessor
from naive_bayes import NaiveBayes
import time
import gc
from collections import defaultdict

def load_sentiment140_dataset(file_path):
    X = []
    y = []
    print(f"Cargando dataset desde {file_path}...")

    with open(file_path, 'r', encoding='latin-1') as file:
        csv_reader = csv.reader(file)
        count = 0
        pos_count = 0
        neg_count = 0

        for row in csv_reader:
            try:
                sentiment = int(row[0].strip('"'))
                text = row[5]

                if sentiment == 0:
                    label = 0
                    neg_count += 1
                elif sentiment == 4:
                    label = 1
                    pos_count += 1
                    if pos_count == 1:
                        print(f"\n¡Primer tweet positivo encontrado! -> {text[:100]}...")
                else:
                    continue

                X.append(text)
                y.append(label)
                count += 1

                if count % 100000 == 0:
                    print(f"Procesados {count} tweets ({pos_count} positivos, {neg_count} negativos)")
                if count == 800000:
                    print(f"\n=== Alcanzados 800,000 registros ===\nNegativos: {neg_count}, Positivos: {pos_count}\n")

            except Exception as e:
                print(f"Error en fila {count}: {e}")
                continue

    print(f"\nTotal cargado: {len(X)}\nPositivos: {pos_count}, Negativos: {neg_count}")
    return X, np.array(y)

def balance_dataset(X_raw, y_raw):
    by_class = defaultdict(list)
    for text, label in zip(X_raw, y_raw):
        by_class[label].append(text)

    min_count = min(len(by_class[0]), len(by_class[1]))
    X_balanced = by_class[0][:min_count] + by_class[1][:min_count]
    y_balanced = [0] * min_count + [1] * min_count

    print(f"\nDataset balanceado: {min_count} negativos y {min_count} positivos")
    return X_balanced, np.array(y_balanced)

def preprocess_texts(X_raw, y_raw, preprocessor):
    X_preprocessed, y_filtered = [], []
    for i, (text, label) in enumerate(zip(X_raw, y_raw)):
        tokens = preprocessor.preprocess(text)
        if tokens:
            X_preprocessed.append(tokens)
            y_filtered.append(label)
        if (i + 1) % 50000 == 0:
            print(f"Preprocesados {i+1}/{len(X_raw)} textos ({(i+1)/len(X_raw)*100:.1f}%)")
            gc.collect()
    return X_preprocessed, np.array(y_filtered)

def main():
    dataset_path = '../dataset/training.1600000.processed.noemoticon.csv'
    os.makedirs('models', exist_ok=True)
    start_time_total = time.time()

    X_raw, y_raw = load_sentiment140_dataset(dataset_path)
    X, y = balance_dataset(X_raw, y_raw)

    print("\nPreprocesando datos...")
    preprocessor = Preprocessor()
    start_pre = time.time()
    X_clean, y_clean = preprocess_texts(X, y, preprocessor)
    print(f"Preprocesamiento completado en {time.time() - start_pre:.2f} segundos")
    print(f"Total después del filtro: {len(X_clean)} ejemplos útiles")

    del X_raw
    gc.collect()

    split = int(len(X_clean) * 0.8)
    X_train, X_test = X_clean[:split], X_clean[split:]
    y_train, y_test = y_clean[:split], y_clean[split:]

    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    print("\nEntrenando modelo Naive Bayes...")
    start_train = time.time()
    model = NaiveBayes()
    model.fit(X_train, y_train)
    print(f"Entrenamiento completado en {time.time() - start_train:.2f} segundos")

    print("\nEvaluando modelo...")
    metrics, confusion = model.get_metrics(X_test, y_test)

    print("\nMétricas de evaluación:")
    for label in metrics:
        title = "Promedios:" if label == 'avg' else f"Clase {label}:"
        print(f"\n{title}")
        for m, v in metrics[label].items():
            print(f"  {m}: {v:.4f}")

    print("\nMatriz de confusión:")
    print(confusion)

    print("\nGuardando archivos...")
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    total = time.time() - start_time_total
    print(f"\nTiempo total de ejecución: {int(total // 60)}m {total % 60:.2f}s")

if __name__ == "__main__":
    main()
