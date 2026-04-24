import streamlit as st
from transformers import pipeline
import torch
import gc

# 1. Configuración de la página
st.set_page_config(page_title="🎬 AI Movie Review Analyzer", page_icon="🎬")

# 2. Funciones de Carga de Modelos
@st.cache_resource
def load_translator(lang):
    try:
        # Usamos la tarea 'translation' a secas para evitar el KeyError
        if lang == "Spanish":
            return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        elif lang == "French":
            return pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    except Exception as e:
        st.error(f"Error técnico en traductor: {e}")
    return None

@st.cache_resource
def load_sentiment_model():
    # Esta es la función que te faltaba definir
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# 3. Interfaz de Usuario
st.title("🎬 Analizador de Reseñas AI")

idioma = st.selectbox("🌐 Idioma original:", ["Spanish", "French", "English"])
texto_usuario = st.text_area("✍️ Escribe tu reseña:")

if st.button("🚀 Analizar Sentimiento"):
    if texto_usuario.strip():
        with st.spinner("La IA está pensando..."):
            texto_final = texto_usuario
            
            # Traducción si no es inglés
            if idioma != "English":
                traductor = load_translator(idioma)
                if traductor:
                    res_trad = traductor(texto_usuario)
                    texto_final = res_trad[0]['translation_text']
                    st.info(f"**En inglés:** {texto_final}")

            # Análisis de Sentimiento
            clasificador = load_sentiment_model()
            resultado = clasificador(texto_final)[0]
            
            # Mostrar Resultados
            if resultado['label'] == "POSITIVE":
                st.success(f"### Sentimiento: POSITIVO 😃 (Confianza: {resultado['score']*100:.2f}%)")
            else:
                st.error(f"### Sentimiento: NEGATIVO 😠 (Confianza: {resultado['score']*100:.2f}%)")
            
            gc.collect()