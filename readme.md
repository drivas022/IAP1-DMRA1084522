# Análisis de Sentimientos en Tweets con Naive Bayes

Este proyecto implementa un clasificador de sentimientos en tweets utilizando el algoritmo Naive Bayes desde cero. Permite clasificar tweets como positivos o negativos, basados en el dataset Sentiment140.

## Estructura del Proyecto

```
IAP1-DMRA1084522/
├── backend/
│   ├── inference.py        # Motor de inferencia para predecir sentimientos
│   ├── naive_bayes.py      # Implementación del algoritmo Naive Bayes
│   ├── preprocessor.py     # Preprocesamiento de texto
│   ├── train_model.py      # Script para entrenar y evaluar el modelo
│   └── models/             # Carpeta donde se guardarán los modelos entrenados
├── dataset/
│   └── training.1600000.processed.noemoticon.csv  # Dataset Sentiment140
└── webapp/
    ├── static/
    │   └── style.css       # Estilos CSS para la interfaz web
    ├── templates/
    │   └── index.html      # Plantilla HTML para la interfaz web
    └── app.py              # Aplicación Flask para servir la interfaz web
```

## Requisitos Previos

Para ejecutar este proyecto, necesitas tener instalado:

1. **Python 3.8 o superior**
   - Verifica tu versión con `python --version`
   - Descarga la última versión desde [python.org](https://www.python.org/downloads/)

2. **Pip (gestor de paquetes de Python)**
   - Normalmente se instala con Python

3. **Dependencias del proyecto**
   ```bash
   pip install flask numpy
   ```

## Configuración Inicial

### 1. Obtener el Dataset

1. Descarga el dataset Sentiment140 desde [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
2. Coloca el archivo CSV descargado en la carpeta `dataset/` con el nombre `training.1600000.processed.noemoticon.csv`

### 2. Preparar el Entorno

1. Clona este repositorio o descarga los archivos
2. Asegúrate de mantener la estructura de carpetas tal como se describió anteriormente
3. Si las carpetas `backend/models/` no existen, créalas:
   ```bash
   mkdir -p backend/models
   ```

## Entrenamiento del Modelo

Existen dos opciones para entrenar el modelo:

### Opción 1: Entrenamiento con Muestra Reducida (Rápido, recomendado para pruebas)

El código por defecto está configurado para usar una muestra de 100,000 tweets, lo que permite un entrenamiento relativamente rápido:

```bash
cd backend
python train_model.py
```

Tiempo estimado: ~3-5 minutos (varía según el hardware).

### Opción 2: Entrenamiento con Dataset Completo (Lento, mejor precisión)

Para entrenar con el dataset completo (1.6 millones de tweets), modifica la línea en `backend/train_model.py`:

```python
# Cambiar esta línea:
X, y = load_sentiment140_dataset(dataset_path, limit=100000)

# Por esta:
X, y = load_sentiment140_dataset(dataset_path)
```

Luego ejecuta:
```bash
cd backend
python train_model.py
```

Tiempo estimado: ~1-3 horas (varía significativamente según el hardware).

**Nota:** El entrenamiento con el dataset completo requiere más memoria RAM. Se recomienda un mínimo de 8GB de RAM para el proceso completo.

## Ejecución de la Aplicación Web

Una vez que el modelo está entrenado y guardado en `backend/models/`:

1. Inicia la aplicación web:
   ```bash
   cd webapp
   python app.py
   ```

2. Abre tu navegador y accede a:
   ```
   http://localhost:5000
   ```

3. Utiliza la interfaz para:
   - Escribir o pegar un tweet para analizar
   - Seleccionar ejemplos del dataset
   - Ver las predicciones de sentimiento (positivo o negativo)
   - Examinar el nivel de confianza y las probabilidades

## Componentes del Proyecto

### Backend

1. **preprocessor.py**
   - Limpieza y tokenización de texto
   - Eliminación de stopwords, URLs, menciones, etc.
   - Construcción de vocabulario

2. **naive_bayes.py**
   - Implementación desde cero del algoritmo Naive Bayes
   - Entrenamiento con suavizado Laplace
   - Cálculo de métricas (precisión, recall, F1-score)

3. **train_model.py**
   - Carga y procesamiento del dataset
   - Entrenamiento y evaluación del modelo
   - Guardado del modelo entrenado

4. **inference.py**
   - Motor de inferencia para nuevos tweets
   - Carga del modelo preentrenado
   - Predicción de sentimientos

### Frontend

1. **app.py**
   - Aplicación Flask para servir la interfaz web
   - Integración con el motor de inferencia
   - Carga de ejemplos del dataset

2. **index.html**
   - Interfaz de usuario para ingresar tweets
   - Visualización de resultados
   - Ejemplos interactivos del dataset

3. **style.css**
   - Estilos responsivos para la interfaz web
   - Visualización de resultados con barras de confianza

## Formato del Dataset

El dataset Sentiment140 contiene tweets etiquetados con su sentimiento:

- **Estructura**: CSV con 6 campos
  1. `target`: 0 (negativo) o 4 (positivo)
  2. `id`: ID del tweet
  3. `date`: Fecha del tweet
  4. `query`: Consulta usada para obtener el tweet (o "NO_QUERY")
  5. `user`: Usuario que publicó el tweet
  6. `text`: Contenido del tweet

- **Ejemplo**:
  ```
  "0","1467810369","Mon Apr 06 22:19:45 PDT 2009","NO_QUERY","_TheSpecialOne_","@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
  ```

## Solución de Problemas

### El modelo no se carga
- Verifica que existan los archivos `backend/models/model.pkl` y `backend/models/preprocessor.pkl`
- Asegúrate de haber ejecutado correctamente `train_model.py`

### Error al cargar el dataset
- Comprueba que el archivo CSV esté en la carpeta correcta (`dataset/`)
- Verifica el nombre del archivo (`training.1600000.processed.noemoticon.csv`)
- Asegúrate de que el archivo no esté dañado

### Problemas de memoria durante el entrenamiento
- Reduce el valor de `limit` en `train_model.py`
- Cierra otras aplicaciones que consuman mucha memoria
- Considera usar un equipo con más RAM para el entrenamiento con el dataset completo

## Evaluación y Métricas

Después del entrenamiento, el sistema mostrará las métricas de evaluación:
- Precisión: exactitud de las predicciones positivas
- Recall: capacidad para encontrar todas las instancias positivas
- F1-score: media armónica entre precisión y recall
- Matriz de confusión: tabla de predicciones correctas e incorrectas

