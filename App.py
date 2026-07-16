import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sistema de Aprobación de Crédito",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
    <style>
    .result-card {
        padding: 1.5rem;
        border-radius: 0.8rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-approved { background-color: #1b3a2b; border: 1px solid #2ecc71; }
    .result-rejected { background-color: #3a1b1b; border: 1px solid #e74c3c; }
    .result-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .result-sub { opacity: 0.8; margin-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    models = joblib.load("models_dict.joblib")
    pca = joblib.load("pca.joblib")
    scaler_pca = joblib.load("scaler.joblib")
    scaler_clf = joblib.load("scaler_clf.joblib")
    return models, pca, scaler_pca, scaler_clf


models, pca, scaler_pca, scaler_clf = load_artifacts()

# Metadata de cada atributo: etiqueta amigable, tipo de control y rango.
FEATURE_INFO = {
    "A1":  {"label": "Género / estado del solicitante", "group": "Datos personales", "type": "bin"},
    "A2":  {"label": "Edad", "group": "Datos personales", "type": "num", "min": 13.0, "max": 80.0, "default": 30.0, "step": 0.5},
    "A4":  {"label": "Estado civil / clasificación demográfica", "group": "Datos personales", "type": "cat", "options": [0, 1, 2, 3]},
    "A5":  {"label": "Dependientes / característica familiar", "group": "Datos personales", "type": "num", "min": 0.0, "max": 15.0, "default": 4.0, "step": 1.0},
    "A9":  {"label": "Posee propiedad / teléfono", "group": "Datos personales", "type": "bin"},

    "A3":  {"label": "Monto de deuda", "group": "Historial financiero", "type": "num", "min": 0.0, "max": 30.0, "default": 2.0, "step": 0.1},
    "A7":  {"label": "Historial / puntuación crediticia", "group": "Historial financiero", "type": "num", "min": 0.0, "max": 30.0, "default": 1.0, "step": 0.1},
    "A8":  {"label": "Indicador de empleo / experiencia", "group": "Historial financiero", "type": "bin"},
    "A10": {"label": "Número de consultas / atributo A10", "group": "Historial financiero", "type": "num", "min": 0.0, "max": 70.0, "default": 0.0, "step": 1.0},
    "A11": {"label": "Indicador adicional A11", "group": "Historial financiero", "type": "bin"},

    "A6":  {"label": "Nivel educacional / situación laboral", "group": "Otros indicadores", "type": "cat", "options": [0, 1, 2, 3, 4]},
    "A12": {"label": "Categoría A12", "group": "Otros indicadores", "type": "cat", "options": [1, 2, 3]},
    "A13": {"label": "Variable numérica A13", "group": "Otros indicadores", "type": "num", "min": 0.0, "max": 2000.0, "default": 100.0, "step": 10.0},
    "A14": {"label": "Variable numérica A14", "group": "Otros indicadores", "type": "num", "min": 0.0, "max": 2500.0, "default": 100.0, "step": 10.0},
}
GROUPS = ["Datos personales", "Historial financiero", "Otros indicadores"]
FEATURE_ORDER = [f"A{i}" for i in range(1, 15)]  # orden con el que se entrenó X

st.title("💳 Sistema de Aprobación de Crédito")
st.caption(
    "Dashboard interactivo para simular la evaluación de una solicitud de tarjeta "
    "de crédito con los modelos entrenados en el notebook (Logistic Regression, "
    "Decision Tree y Random Forest)."
)

# ---------- Sidebar: identificación + selección de modelo ----------
st.sidebar.header("Solicitud")
customer_id = st.sidebar.number_input("ID del cliente", value=1000, step=1)
model_name = st.sidebar.selectbox("Modelo a usar", list(models.keys()))
selected_model = models[model_name]

st.sidebar.divider()
st.sidebar.header("Datos del solicitante")

input_data = {}
for group in GROUPS:
    with st.sidebar.expander(group, expanded=(group == "Datos personales")):
        for feat, info in FEATURE_INFO.items():
            if info["group"] != group:
                continue
            if info["type"] == "bin":
                input_data[feat] = 1.0 if st.checkbox(info["label"], key=feat) else 0.0
            elif info["type"] == "cat":
                input_data[feat] = float(st.selectbox(info["label"], info["options"], key=feat))
            else:
                input_data[feat] = st.slider(
                    info["label"], info["min"], info["max"], info["default"], info["step"], key=feat
                )

input_df = pd.DataFrame([input_data])[FEATURE_ORDER]

tab_eval, tab_compare = st.tabs(["🔎 Evaluación individual", "📊 Comparar modelos"])

with tab_eval:
    col_btn, _ = st.columns([1, 3])
    evaluate = col_btn.button("Evaluar solicitud", type="primary", use_container_width=True)

    if evaluate:
        scaled_for_model = scaler_clf.transform(input_df)
        prediction = selected_model.predict(scaled_for_model)[0]
        proba = None
        if hasattr(selected_model, "predict_proba"):
            proba = selected_model.predict_proba(scaled_for_model)[0]

        approved = prediction == 1
        card_class = "result-approved" if approved else "result-rejected"
        icon = "✅" if approved else "❌"
        text = "Solicitud Aprobada" if approved else "Solicitud Rechazada"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <p class="result-title">{icon} {text}</p>
                <p class="result-sub">Modelo utilizado: {model_name}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if proba is not None:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=proba[1] * 100,
                        number={"suffix": "%"},
                        title={"text": "Confianza de aprobación"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#2ecc71" if approved else "#e74c3c"},
                            "steps": [
                                {"range": [0, 50], "color": "#3a1b1b"},
                                {"range": [50, 100], "color": "#1b3a2b"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Este modelo no entrega probabilidad, solo clase predicha.")

        with col2:
            st.subheader("Ubicación en el espacio de clusters (PCA)")
            pca_input = pd.DataFrame([{"CustomerID": customer_id, **input_data}])[
                ["CustomerID"] + FEATURE_ORDER
            ]
            pca_coords = pca.transform(scaler_pca.transform(pca_input))[0]
            fig_pca = go.Figure(
                go.Scatter(
                    x=[pca_coords[0]],
                    y=[pca_coords[1]],
                    mode="markers",
                    marker=dict(size=16, color="#3498db"),
                    name="Solicitante",
                )
            )
            fig_pca.update_layout(
                height=280,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis_title="Componente 1",
                yaxis_title="Componente 2",
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with st.expander("Ver datos ingresados"):
            st.dataframe(input_df, use_container_width=True)
    else:
        st.info("Completa los datos en la barra lateral y presiona **Evaluar solicitud**.")

with tab_compare:
    st.subheader("Comparación de los 3 modelos para la misma solicitud")
    scaled_for_model = scaler_clf.transform(input_df)

    rows = []
    for name, model in models.items():
        pred = model.predict(scaled_for_model)[0]
        conf = model.predict_proba(scaled_for_model)[0][1] if hasattr(model, "predict_proba") else None
        rows.append(
            {
                "Modelo": name,
                "Resultado": "✅ Aprobada" if pred == 1 else "❌ Rechazada",
                "Confianza aprobación": f"{conf * 100:.1f}%" if conf is not None else "N/A",
            }
        )
    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    conf_values = [
        models[n].predict_proba(scaled_for_model)[0][1] * 100
        if hasattr(models[n], "predict_proba")
        else 0
        for n in models
    ]
    fig_bar = go.Figure(
        go.Bar(
            x=list(models.keys()),
            y=conf_values,
            marker_color=["#3498db", "#9b59b6", "#2ecc71"],
            text=[f"{v:.1f}%" for v in conf_values],
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        yaxis_title="Confianza de aprobación (%)",
        yaxis_range=[0, 100],
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
