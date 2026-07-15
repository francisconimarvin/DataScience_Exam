import streamlit as st
import pandas as pd
import joblib

# Carga de artefactos
models = joblib.load('models_dict.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')

st.title("💳 Sistema de Aprobación de Crédito")

# Selección de modelo para la predicción
model_name = st.selectbox("Selecciona el modelo de predicción:", list(models.keys()))
selected_model = models[model_name]

# Inputs del usuario (A1-A14)
input_data = {f'A{i}': st.sidebar.slider(f'Atributo A{i}', 0.0, 10.0, 5.0) for i in range(1, 15)}
input_df = pd.DataFrame(input_data, index=[0])

if st.button("Evaluar Solicitud"):
    # Escalado (usando el mismo objeto que entrenaste)
    scaled_data = scaler.transform(input_df)
    
    # Predicción
    prediction = selected_model.predict(scaled_data)
    
    resultado = "✅ Aprobada" if prediction[0] == 1 else "❌ Rechazada"
    st.subheader(f"Resultado con {model_name}: {resultado}")
    
    # Visualización PCA
    pca_data = pca.transform(scaled_data)
    st.write(f"Coordenadas en espacio PCA: {pca_data[0]}")