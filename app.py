import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="IA Pesquera Cantábrico", layout="centered")
st.title("IA Pesquera del Cantábrico 🌊🐙")

# ----------------------------------------
# Cargar dataset mejorado
# ----------------------------------------
df = pd.read_csv("dataset_pesca_cantabrico_mejorado.csv")

# Variables predictoras
features = ["temp_agua","oleaje","viento","lluvia","presion","coef_marea",
            "temp_superficie","salinidad","oxigeno_disuelto","ph_agua"]

# Etiqueta: especie con mayor captura simulada
df["especie"] = df[["captura_pulpo","captura_lubina","captura_percebe"]].idxmax(axis=1)

X = df[features]
y = df["especie"]

# Entrenar modelo de IA
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

st.markdown("### Ajusta las condiciones ambientales y tecnológicas para predecir la mejor especie a capturar")

# ----------------------------------------
# Sliders interactivos
# ----------------------------------------
temp = st.slider("Temperatura agua (ºC)", float(df["temp_agua"].min()), float(df["temp_agua"].max()), float(df["temp_agua"].mean()))
ola = st.slider("Oleaje (m)", float(df["oleaje"].min()), float(df["oleaje"].max()), float(df["oleaje"].mean()))
viento = st.slider("Viento (km/h)", float(df["viento"].min()), float(df["viento"].max()), float(df["viento"].mean()))
lluvia = st.slider("Lluvia (mm)", float(df["lluvia"].min()), float(df["lluvia"].max()), float(df["lluvia"].mean()))
presion = st.slider("Presión atmosférica (hPa)", float(df["presion"].min()), float(df["presion"].max()), float(df["presion"].mean()))
marea = st.slider("Coeficiente de marea", int(df["coef_marea"].min()), int(df["coef_marea"].max()), int(df["coef_marea"].mean()))
temp_sup = st.slider("Temperatura superficial (ºC)", float(df["temp_superficie"].min()), float(df["temp_superficie"].max()), float(df["temp_superficie"].mean()))
sal = st.slider("Salinidad (‰)", float(df["salinidad"].min()), float(df["salinidad"].max()), float(df["salinidad"].mean()))
ox = st.slider("Oxígeno disuelto (mg/L)", float(df["oxigeno_disuelto"].min()), float(df["oxigeno_disuelto"].max()), float(df["oxigeno_disuelto"].mean()))
ph = st.slider("pH del agua", float(df["ph_agua"].min()), float(df["ph_agua"].max()), float(df["ph_agua"].mean()))

# ----------------------------------------
# Predicción
# ----------------------------------------
if st.button("Predecir pesca"):
    pred = modelo.predict([[temp, ola, viento, lluvia, presion, marea,
                            temp_sup, sal, ox, ph]])
    st.success(f"✅ Mejor opción hoy: {pred[0]}")

st.markdown("---")
st.markdown("### Primeros registros del dataset")
st.dataframe(df.head(10))
