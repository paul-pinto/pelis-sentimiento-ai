import streamlit as st
from transformers import pipeline
import torch
import gc

# 1. Configuración de la página
st.set_page_config(
    page_title="🎬 AI Movie Review Analyzer",
    page_icon="🎬",
    layout="centered"
)

# 2. Funciones con Caché para optimizar RAM
# Usamos @st.cache_resource para que los modelos se carguen solo una vez
@st.cache_resource
def load_translator(lang):
    try:
        if lang == "Spanish":
            # Usamos la tarea genérica 'translation' pero con el modelo específico
            return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        elif lang == "French":
            return pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    except Exception as e:
        st.error(f"Error al cargar el traductor: {e}")
    return None

@st.cache_resource
def load_translator(lang):
    try:
        if lang == "Spanish":
            # Usamos la tarea genérica 'translation'
            return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        elif lang == "French":
            return pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    except Exception as e:
        st.error(f"Error al cargar el traductor: {e}")
    return None

# 3. Interfaz de Usuario (UI)
st.title("🎬 Analizador de Reseñas AI")
st.markdown("""
Esta aplicación utiliza modelos de **Deep Learning** para traducir reseñas y analizar 
si el sentimiento es **Positivo** o **Negativo**. 
""")

st.divider()

# Selector de idioma y entrada de texto
idioma = st.selectbox("🌐 Selecciona el idioma original de la reseña:", ["Spanish", "French", "English"])
texto_usuario = st.text_area("✍️ Escribe tu reseña aquí:", placeholder="Ej: La película fue increíble, me encantó la fotografía...", height=150)

if st.button("🚀 Analizar Sentimiento", type="primary"):
    if texto_usuario.strip():
        with st.spinner("Procesando con Inteligencia Artificial..."):
            try:
                texto_para_analizar = texto_usuario
                
                # --- PASO 1: TRADUCCIÓN ---
                if idioma != "English":
                    traductor = load_translator(idioma)
                    if traductor:
                        resultado_trad = traductor(texto_usuario)
                        texto_para_analizar = resultado_trad[0]['translation_text']
                        st.info(f"**Traducción automática (EN):** {texto_para_analizar}")
                
                # --- PASO 2: ANÁLISIS DE SENTIMIENTO ---
                clasificador = load_sentiment_model()
                resultado_sent = clasificador(texto_para_analizar)[0]
                
                label = resultado_sent['label']
                score = resultado_sent['score']
                
                # --- PASO 3: MOSTRAR RESULTADOS ---
                st.subheader("Resultado del Análisis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if label == "POSITIVE":
                        st.success(f"### Sentimiento: POSITIVO 😃")
                    else:
                        st.error(f"### Sentimiento: NEGATIVO 😠")
                
                with col2:
                    st.metric("Nivel de Confianza", f"{score*100:.2f}%")
                
                # Limpieza de memoria manual tras el proceso
                gc.collect()

            except Exception as e:
                st.error(f"Ocurrió un error durante el procesamiento: {e}")
    else:
        st.warning("⚠️ Por favor, ingresa un texto antes de analizar.")

# Pie de página
st.divider()
st.caption("Desarrollado con Streamlit y Hugging Face Transformers.")