import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer
import torch
import gc

# 1. Configuración de la página
st.set_page_config(
    page_title="🎬 AI Movie Review Analyzer", 
    page_icon="🎬",
    layout="centered"
)

# 2. Funciones de Carga de Modelos con Caché
@st.cache_resource
def load_translator(lang):
    """
    Carga manualmente el modelo y tokenizador de Helsinki-NLP para evitar 
    errores de registro de tareas en la nube.
    """
    try:
        if lang == "Spanish":
            model_name = "Helsinki-NLP/opus-mt-es-en"
        elif lang == "French":
            model_name = "Helsinki-NLP/opus-mt-fr-en"
        else:
            return None
            
        # Cargamos componentes manualmente para mayor estabilidad
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Creamos una función interna que emula el comportamiento del pipeline
        def translate_func(text):
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            decoded_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            return [{"translation_text": decoded_text}]
            
        return translate_func
    except Exception as e:
        st.error(f"Error técnico en el motor de traducción: {e}")
    return None

@st.cache_resource
def load_sentiment_model():
    """
    Carga el modelo de análisis de sentimiento DistilBERT.
    """
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# 3. Interfaz de Usuario (UI)
st.title("🎬 Analizador de Reseñas AI")
st.markdown("Analiza el sentimiento de tus películas favoritas en **Español, Francés o Inglés**.")

st.divider()

# Entradas del usuario
idioma = st.selectbox("🌐 Selecciona el idioma original:", ["Spanish", "French", "English"])
texto_usuario = st.text_area("✍️ Escribe tu reseña aquí:", height=150)

if st.button("🚀 Analizar Sentimiento", type="primary"):
    if texto_usuario.strip():
        with st.spinner("La Inteligencia Artificial está procesando..."):
            try:
                texto_final = texto_usuario
                
                # --- FASE 1: TRADUCCIÓN ---
                if idioma != "English":
                    traductor = load_translator(idioma)
                    if traductor:
                        resultado_trad = traductor(texto_usuario)
                        texto_final = resultado_trad[0]['translation_text']
                        st.info(f"**Traducción al Inglés:** {texto_final}")

                # --- FASE 2: ANÁLISIS DE SENTIMIENTO ---
                clasificador = load_sentiment_model()
                resultado = clasificador(texto_final)[0]
                
                label = resultado['label']
                confianza = resultado['score'] * 100

                # --- FASE 3: MOSTRAR RESULTADOS ---
                st.subheader("Resultado")
                if label == "POSITIVE":
                    st.success(f"### Sentimiento: POSITIVO 😃")
                    st.balloons()
                else:
                    st.error(f"### Sentimiento: NEGATIVO 😠")
                
                st.write(f"**Nivel de confianza:** {confianza:.2f}%")
                
                # Liberar memoria RAM
                gc.collect()

            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
    else:
        st.warning("⚠️ Por favor, ingresa un texto para analizar.")

st.divider()
st.caption("Proyecto de Maestría en Data Science & IA - Jhonny Paul Pinto Phillips")