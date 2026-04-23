🎬 Movie Review AI: Translation & Sentiment Analysis

Este proyecto realiza un pipeline completo de Procesamiento de Lenguaje Natural (NLP) para unificar reseñas de películas en múltiples idiomas y analizar su sentimiento utilizando modelos de Transformers de última generación.
📋 Descripción del Proyecto

El sistema procesa un conjunto de datos multilingüe (Inglés, Francés y Español). A través de un flujo automatizado, los textos son traducidos al inglés y posteriormente clasificados según su carga emocional (Positiva o Negativa).

Características principales:

    Ingesta Multilingüe: Procesamiento de archivos CSV con esquemas de datos en diferentes idiomas.

    Traducción de Alta Precisión: Uso de modelos Helsinki-NLP para traducciones del francés y español al inglés.

    Análisis de Sentimiento: Clasificación mediante el modelo DistilBERT optimizado para análisis de texto.

    Arquitectura Robusta: Implementación de técnicas de ahorro de memoria (Garbage Collection) y gestión de hilos para ejecución en hardware con recursos limitados.

🛠️ Tecnologías Utilizadas

    Python 3.11+

    Pandas: Manipulación y limpieza de datos.

    Hugging Face Transformers: Inferencia de modelos pre-entrenados.

    PyTorch: Motor de ejecución para los modelos de Deep Learning.

    SQLite: Persistencia de resultados en una base de datos relacional.

    Streamlit: Interfaz web interactiva para usuarios finales.

🚀 Instalación y Uso

    Clonar el repositorio:
    Bash

    git clone https://github.com/paul-pinto/pelis-sentimiento-ai.git
    cd pelis-sentimiento-ai

    Crear y activar el entorno virtual:
    Bash

    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate

    Instalar dependencias:
    Bash

    pip install -r requirements.txt

    Ejecutar la App:
    Bash

    streamlit run app.py

📊 Estructura de Salida

El proceso genera un archivo final reviews_with_sentiment.csv con el siguiente esquema:

    Title: Nombre de la obra.

    Year: Año de estreno.

    Synopsis: Sinopsis traducida al inglés.

    Review: Reseña traducida al inglés.

    Original_Language: Idioma origen antes del procesamiento.

    Sentiment: Resultado del análisis (POSITIVE / NEGATIVE).

🧪 Evaluación de Rendimiento

El modelo fue evaluado mediante una muestra controlada, obteniendo métricas de precisión y exactitud mediante scikit-learn, asegurando la fiabilidad de las inferencias realizadas por los Transformers.

Desarrollado por Jhonny Paul Pinto Phillips
Candidato a Máster en Data Science e IA