import streamlit as st
from transformers import pipeline

# Configuración de la página
st.set_page_config(page_title="AI Movie Review", page_icon="🎬")

# Caché para que los modelos se carguen solo una vez y no colapsen la RAM
@st.cache_resource
def load_translator(lang):
    if lang == "Spanish":
        return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
    elif lang == "French":
        return pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    return None

@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Interfaz de Usuario
st.title("🎬 Analizador de Reseñas AI")
st.write("Escribe una reseña de una película en tu idioma y deja que la Inteligencia Artificial haga el resto.")

idioma = st.selectbox("¿En qué idioma está tu reseña?", ["Spanish", "French", "English"])
texto = st.text_area("Escribe o pega la reseña aquí:", height=150)

if st.button("Analizar Reseña", type="primary"):
    if texto.strip():
        with st.spinner("La IA está pensando... 🧠"):
            texto_ingles = texto
            
            # 1. Traducción (si es necesario)
            if idioma != "English":
                traductor = load_translator(idioma)
                resultado_trad = traductor(texto)
                texto_ingles = resultado_trad[0]['translation_text']
                st.info(f"**Traducción automática:** {texto_ingles}")
            
            # 2. Análisis de Sentimiento
            clasificador = load_sentiment_model()
            resultado_sent = clasificador(texto_ingles)[0]
            
            label = resultado_sent['label']
            score = resultado_sent['score']
            
            # 3. Mostrar Resultados
            st.divider()
            if label == "POSITIVE":
                st.success(f"### Sentimiento: POSITIVO 😃\n**Nivel de confianza:** {score*100:.2f}%")
            else:
                st.error(f"### Sentimiento: NEGATIVO 😠\n**Nivel de confianza:** {score*100:.2f}%")
    else:
        st.warning("Por favor, ingresa un texto para analizar.")