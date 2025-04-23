import csv
import os

def debug_dataset(file_path):
    """
    Debug para verificar la estructura del dataset
    """
    print(f"Verificando dataset en: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: El archivo no existe en {file_path}")
        return
    
    try:
        with open(file_path, 'r', encoding='latin-1') as file:
            # Leer las primeras 10 líneas con diferentes métodos
            print("\n=== Leyendo con csv.reader ===")
            csv_reader = csv.reader(file)
            for i, row in enumerate(csv_reader):
                if i < 5:
                    print(f"Fila {i}: {row}")
                    print(f"  - Sentimiento (col 0): '{row[0]}'")
                    print(f"  - Texto (col 5): '{row[5]}'")
                else:
                    break
            
            # Reiniciar el archivo
            file.seek(0)
            
            print("\n=== Leyendo línea por línea ===")
            for i, line in enumerate(file):
                if i < 5:
                    print(f"Línea {i}: {line[:100]}...")
                else:
                    break
            
            # Reiniciar el archivo y contar positivos y negativos
            file.seek(0)
            csv_reader = csv.reader(file)
            
            pos_count = 0
            neg_count = 0
            error_count = 0
            
            for i, row in enumerate(csv_reader):
                if i >= 100000:  # Solo verificar los primeros 100,000
                    break
                    
                try:
                    sentiment = int(row[0].strip('"'))
                    if sentiment == 0:
                        neg_count += 1
                    elif sentiment == 4:
                        pos_count += 1
                except Exception as e:
                    error_count += 1
                    if error_count < 5:
                        print(f"Error en fila {i}: {e}")
                        print(f"Fila completa: {row}")
            
            print(f"\nResultados de los primeros 100,000 registros:")
            print(f"Negativos (0): {neg_count}")
            print(f"Positivos (4): {pos_count}")
            print(f"Errores: {error_count}")
            
    except Exception as e:
        print(f"Error general: {e}")

if __name__ == "__main__":
    # Usa la misma ruta que en tu train_model.py
    dataset_path = '../dataset/training.1600000.processed.noemoticon.csv'
    debug_dataset(dataset_path)