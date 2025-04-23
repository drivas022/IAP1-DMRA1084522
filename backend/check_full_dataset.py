import csv
import os
import sys

def check_full_dataset():
    """
    Verifica que se lean todos los 1.6 millones de tweets del dataset
    """
    dataset_path = '../dataset/training.1600000.processed.noemoticon.csv'
    
    print(f"Verificando dataset completo en: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: El archivo no existe en {dataset_path}")
        return
    
    try:
        with open(dataset_path, 'r', encoding='latin-1') as file:
            csv_reader = csv.reader(file)
            
            total_count = 0
            pos_count = 0
            neg_count = 0
            first_positive_row = None
            last_negative_row = None
            
            print("Leyendo todo el dataset... Esto puede tomar unos minutos.")
            
            for i, row in enumerate(csv_reader):
                total_count += 1
                
                try:
                    sentiment = int(row[0].strip('"'))
                    
                    if sentiment == 0:
                        neg_count += 1
                        last_negative_row = i
                    elif sentiment == 4:
                        pos_count += 1
                        if first_positive_row is None:
                            first_positive_row = i
                            print(f"\n¡Primer tweet positivo encontrado en la fila {i}!")
                            print(f"Sentimiento: {sentiment}")
                            print(f"Tweet: {row[5][:100]}...")
                    
                    # Mostrar progreso cada 100,000 registros
                    if (i + 1) % 100000 == 0:
                        print(f"Procesados {i+1:,} registros: {neg_count:,} negativos, {pos_count:,} positivos")
                    
                    # Mostrar información en el punto de transición
                    if i == 800000:
                        print(f"\n=== Llegamos a la fila 800,000 ===")
                        print(f"Negativos hasta ahora: {neg_count}")
                        print(f"Positivos hasta ahora: {pos_count}")
                        print("Continuando...\n")
                    
                except Exception as e:
                    print(f"Error en la fila {i}: {e}")
            
            # Resumen final
            print("\n" + "="*50)
            print("RESUMEN FINAL DEL DATASET")
            print("="*50)
            print(f"Total de registros: {total_count:,}")
            print(f"Tweets negativos: {neg_count:,}")
            print(f"Tweets positivos: {pos_count:,}")
            print(f"Proporción: {neg_count/total_count:.1%} negativos, {pos_count/total_count:.1%} positivos")
            if first_positive_row:
                print(f"Primer tweet positivo encontrado en la fila: {first_positive_row:,}")
            else:
                print("¡No se encontraron tweets positivos!")
            if last_negative_row:
                print(f"Último tweet negativo encontrado en la fila: {last_negative_row:,}")
            print("="*50)
            
            # Verificar si el dataset está completo
            if total_count < 1600000:
                print(f"\n¡ADVERTENCIA! El dataset está incompleto. Se esperaban 1,600,000 registros pero solo se encontraron {total_count:,}")
            elif pos_count == 0:
                print(f"\n¡ADVERTENCIA! No se encontraron tweets positivos en el dataset.")
            elif pos_count < 700000 or neg_count < 700000:
                print(f"\n¡ADVERTENCIA! La distribución de clases parece incorrecta.")
            else:
                print(f"\n✓ El dataset parece estar completo y correctamente distribuido.")
            
    except Exception as e:
        print(f"Error general: {e}")

if __name__ == "__main__":
    check_full_dataset()