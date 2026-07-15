import streamlit as st
import pandas as pd
import joblib

# 1. Carga de artefactos
models = joblib.load('models_dict.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')

st.title("💳 Sistema de Aprobación de Crédito")

# 2. Selección de modelo
model_name = st.selectbox("Selecciona el modelo:", list(models.keys()))
selected_model = models[model_name]

# 3. Inputs
st.sidebar.header("Datos del Solicitante")
input_data = {'CustomerID': st.sidebar.number_input("ID del Cliente", value=1000)}
for i in range(1, 15):
    input_data[f'A{i}'] = st.sidebar.slider(f'Atributo A{i}', 0.0, 10.0, 5.0, key=f"s{i}")

# 4. Preparación (Alineación para el Scaler)
feature_names = ['CustomerID'] + [f'A{i}' for i in range(1, 15)]
input_df = pd.DataFrame(input_data, index=[0])[feature_names]

if st.button("Evaluar Solicitud"):
    # PASO A: Escalar (el scaler espera las 15)
    scaled_data = scaler.transform(input_df)
    
    # PASO B: Filtrar solo las 14 columnas para modelos como LogReg
    # Si el modelo tiene un método 'n_features_in_', lo usamos para verificar
    if hasattr(selected_model, 'n_features_in_') and selected_model.n_features_in_ == 14:
        # Esto elimina la primera columna (CustomerID) que está en la posición 0
        scaled_data_for_model = scaled_data[:, 1:] 
    else:
        scaled_data_for_model = scaled_data

    # PASO C: Predicción con los datos correctos
    prediction = selected_model.predict(scaled_data_for_model)
    
    resultado = "✅ Aprobada" if prediction[0] == 1 else "❌ Rechazada"
    st.subheader(f"Resultado con {model_name}: {resultado}")
    
    # PCA usa el escalado original (15 columnas)
    pca_data = pca.transform(scaled_data)
    st.write(f"Coordenadas PCA: {pca_data[0][0]:.2f}, {pca_data[0][1]:.2f}")