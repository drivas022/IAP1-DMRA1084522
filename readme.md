# Enlace del sitio

¡Ingresa [aquí](https://analisis-de-sentimientos-en-tweets-con.onrender.com) para visualizar nuestra página web!.

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

El código está configurado para entrenar el modelo con el dataset completo, procesando hasta 800,000 tweets (balanceando entre positivos y negativos):

```bash
cd backend
python train_model.py
```

El proceso incluye:
1. Carga del dataset completo
2. Balanceo de clases (igual cantidad de tweets positivos y negativos)
3. Preprocesamiento y tokenización
4. División en conjuntos de entrenamiento (80%) y prueba (20%)
5. Entrenamiento del modelo Naive Bayes
6. Evaluación con métricas detalladas
7. Guardado del modelo y preprocessor

Tiempo estimado: ~15-30 minutos (varía según el hardware).

**Nota:** El entrenamiento requiere suficiente memoria RAM. Se recomienda un mínimo de 8GB de RAM para el proceso completo.

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
   - Seleccionar ejemplos reales del dataset
   - Ver las predicciones de sentimiento (positivo o negativo)
   - Examinar el nivel de confianza y las probabilidades detalladas

## Componentes del Proyecto

### Backend

1. **preprocessor.py**
   - Limpieza completa de texto (minúsculas, eliminación de URLs, menciones, hashtags, RT, números)
   - Eliminación de puntuación y caracteres especiales
   - Tokenización y eliminación de stopwords
   - Construcción de vocabulario con filtrado por frecuencia mínima

2. **naive_bayes.py**
   - Implementación desde cero del algoritmo Naive Bayes
   - Suavizado Laplace (alpha=1.0) para manejar palabras desconocidas
   - Cálculo de probabilidades en escala logarítmica para estabilidad numérica
   - Métricas de evaluación completas (precisión, recall, F1-score)
   - Matriz de confusión para análisis detallado

3. **train_model.py**
   - Carga del dataset con manejo de encoding latin-1
   - Balanceo automático de clases
   - Preprocesamiento optimizado con recolección de basura
   - Guardado de modelo y preprocessor usando pickle

4. **inference.py**
   - Motor de inferencia para nuevos tweets
   - Carga del modelo y preprocessor preentrenados
   - Mapeo de etiquetas numéricas a texto legible
   - Predicción con probabilidades detalladas

### Frontend

1. **app.py**
   - Aplicación Flask con rutas específicas para análisis y estado
   - Integración directa con el motor de inferencia
   - Carga dinámica de ejemplos reales desde el dataset
   - Manejo robusto de errores y validaciones

2. **index.html**
   - Interfaz moderna y responsiva
   - Visualización clara de resultados con íconos de sentimiento
   - Barras de progreso animadas para probabilidades
   - Sección de ejemplos interactivos del dataset real

3. **style.css**
   - Diseño responsivo con media queries
   - Variables CSS para consistencia de colores
   - Efectos visuales y animaciones para mejor UX
   - Soporte para temas claros y oscuros

## Formato del Dataset

El dataset Sentiment140 contiene 1.6 millones de tweets etiquetados:

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
- Comprueba que las rutas relativas sean correctas en `webapp/app.py`

### Error al cargar el dataset
- Comprueba que el archivo CSV esté en la carpeta correcta (`dataset/`)
- Verifica el nombre del archivo (`training.1600000.processed.noemoticon.csv`)
- Asegúrate de que el archivo no esté dañado y sea legible

### Problemas de memoria durante el entrenamiento
- El código incluye manejo de memoria con `gc.collect()`
- Si persisten los problemas, considera usar una máquina con más RAM
- Procesa el dataset en lotes más pequeños si es necesario

### La interfaz web no carga ejemplos
- Verifica que el dataset esté en la ubicación correcta
- Asegúrate de que Flask pueda acceder al archivo del dataset
- Revisa los permisos de lectura del archivo CSV

## Evaluación y Métricas

El sistema proporciona evaluación completa con:
- **Precisión**: Exactitud de las predicciones positivas
- **Recall**: Capacidad para encontrar todas las instancias positivas
- **F1-score**: Media armónica entre precisión y recall
- **Matriz de confusión**: Análisis detallado de predicciones correctas e incorrectas
- **Métricas por clase y promedio**: Evaluación detallada para cada sentimiento

Las métricas se calculan automáticamente al finalizar el entrenamiento y se muestran en la consola.
