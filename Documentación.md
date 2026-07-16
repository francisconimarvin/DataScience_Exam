# Informe Técnico: Sistema de Aprobación de Crédito

## 1. Resumen Ejecutivo
El presente proyecto consiste en el desarrollo de un sistema inteligente diseñado para la evaluación automática de solicitudes de tarjetas de crédito. La solución integra modelos de *Machine Learning* supervisado (Regresión Logística, Árboles de Decisión y Bosques Aleatorios) con una interfaz web interactiva desarrollada en **Streamlit**. El sistema permite a los usuarios realizar simulaciones de crédito en tiempo real, comparar el rendimiento de diversos algoritmos y visualizar la posición relativa de los solicitantes mediante técnicas de reducción de dimensionalidad (PCA).

## 2. Arquitectura del Sistema
El flujo de trabajo del proyecto se divide en dos capas fundamentales:

* **Capa de Entrenamiento (Notebook):**
    * Preprocesamiento de datos y manejo de variables (escalado y limpieza).
    * Entrenamiento de modelos y validación.
    * Reducción de dimensionalidad mediante **PCA** (Componentes Principales) para análisis de clusters.
    * Serialización de artefactos mediante `joblib` (`models_dict.joblib`, `pca.joblib`, `scaler.joblib`, `scaler_clf.joblib`).
* **Capa de Interfaz (App.py):**
    * Dashboard interactivo que carga los modelos serializados.
    * Gestión de entradas de usuario mediante formularios dinámicos y *sliders*.
    * Inferencia en tiempo real y visualización de resultados mediante `plotly`.

## 3. Modelos y Metodología
Se implementaron tres enfoques para la clasificación del riesgo crediticio:

| Modelo | Descripción | Propósito |
| :--- | :--- | :--- |
| **Regresión Logística** | Clasificador lineal basado en probabilidad. | Modelo base (baseline) para interpretar coeficientes. |
| **Árbol de Decisión** | Clasificador basado en particiones jerárquicas. | Capturar reglas de decisión no lineales. |
| **Random Forest** | *Ensemble* de múltiples árboles de decisión. | Maximizar la robustez y precisión mediante votación. |

## 4. Lógica de Preprocesamiento y Validación
El sistema asegura la consistencia de los datos entre el entrenamiento y la predicción:
* **Alineación de Features:** Se utiliza `FEATURE_ORDER` para garantizar que el orden de las 14 variables de entrada (`A1` a `A14`) sea idéntico al utilizado durante el entrenamiento.
* **Escalamiento Independiente:** Se aplican escaladores específicos para el modelo de clasificación (`scaler_clf`) y para la visualización PCA (`scaler_pca`), evitando la contaminación de datos (*data leakage*).
* **Manejo de Errores:** Se implementó una lógica de carga con caché (`@st.cache_resource`) para optimizar el rendimiento y evitar la recarga redundante de archivos en memoria.

## 5. Guía de Ejecución
Para levantar el sistema en cualquier entorno local, asegúrese de tener instaladas las dependencias y seguir estos pasos:

1. **Entorno:** Asegúrese de estar en el entorno virtual (`.venv`) configurado.
2. **Dependencias:**
   ```bash
   pip install streamlit pandas joblib plotly scikit-learn
Archivos: Verifique que todos los archivos **.joblib** se encuentren en la raíz del proyecto junto a App.py.

Lanzamiento: Ejecute el servidor de Streamlit:
Para lanzar el dashboard, dentro de la raíz del proyecto, se hace
```bash 
  streamlit run App.py
```
## 6. Conclusiones

La implementación del dashboard facilita la interpretabilidad de los modelos de Machine Learning en un contexto financiero. 
La combinación de predicciones discretas con probabilidades de aprobación y la visualización de datos en un espacio de componentes principales (PCA) convierte a esta herramienta en un recurso robusto para el análisis de riesgo crediticio, cumpliendo con los estándares de ciencia de datos aplicados.
