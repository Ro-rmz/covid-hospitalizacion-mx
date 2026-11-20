"""
App web (Streamlit) que consulta el modelo publicado en Databricks.

Ejecutar:  streamlit run app/app.py
"""
import json
import os
import time

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from cliente_modelo import ClienteModelo, EntradaPaciente, ErrorModelo

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ENTIDADES = [
    "Aguascalientes", "Baja California", "Baja California Sur", "Campeche", "Coahuila", "Colima",
    "Chiapas", "Chihuahua", "Ciudad de México", "Durango", "Guanajuato", "Guerrero", "Hidalgo",
    "Jalisco", "Estado de México", "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca",
    "Puebla", "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora", "Tabasco",
    "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas",
]
COMORBILIDADES = {
    "diabetes": "Diabetes",
    "hipertension": "Hipertensión",
    "obesidad": "Obesidad",
    "epoc": "EPOC",
    "asma": "Asma",
    "inmunosupresion": "Inmunosupresión",
    "cardiovascular": "Enfermedad cardiovascular",
    "renal_cronica": "Enfermedad renal crónica",
    "tabaquismo": "Tabaquismo",
    "otra_comorbilidad": "Otra comorbilidad",
}
COLOR = {"bajo": "🟢", "medio": "🟡", "alto": "🔴"}

st.set_page_config(page_title="Riesgo de hospitalización COVID-19", page_icon="🏥", layout="centered")


@st.cache_resource
def obtener_cliente():
    return ClienteModelo(
        host=os.getenv("DATABRICKS_HOST", ""),
        token=os.getenv("DATABRICKS_TOKEN", ""),
        endpoint=os.getenv("ENDPOINT_NAME", "covid-hospitalizacion"),
        timeout_lectura=float(os.getenv("TIMEOUT_SEGUNDOS", "60")),
        reintentos=int(os.getenv("REINTENTOS", "3")),
    )


st.title("🏥 Riesgo de hospitalización por COVID-19")
st.caption(
    "Modelo entrenado con millones de casos confirmados de la Secretaría de Salud (Datos Abiertos DGE), "
    "procesados con arquitectura Medallion en Databricks y publicado como endpoint REST."
)

try:
    cliente = obtener_cliente()
except ErrorModelo as e:
    st.error(f"⚙️ Configuración incompleta: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Estado del servicio")
    estado = cliente.estado()
    if estado == "READY":
        st.success(f"Endpoint `{cliente.endpoint}`: READY")
    else:
        st.warning(f"Endpoint `{cliente.endpoint}`: {estado}")
    st.caption(f"URL: {cliente.url_invocacion}")
    st.divider()
    st.caption("⚠️ Herramienta académica. No sustituye una valoración médica.")

with st.form("paciente"):
    c1, c2 = st.columns(2)
    with c1:
        sexo = st.radio("Sexo", ["Mujer", "Hombre"], horizontal=True)
        edad = st.number_input("Edad (años)", min_value=0, max_value=110, value=45, step=1)
    with c2:
        entidad = st.selectbox("Entidad de residencia", ENTIDADES, index=ENTIDADES.index("Jalisco"))
        embarazo = st.checkbox("Embarazo", disabled=(sexo == "Hombre"))

    st.markdown("**Comorbilidades**")
    cols = st.columns(2)
    marcadas = {
        clave: cols[i % 2].checkbox(etiqueta, key=clave)
        for i, (clave, etiqueta) in enumerate(COMORBILIDADES.items())
    }
    enviar = st.form_submit_button("Calcular riesgo", type="primary", use_container_width=True)

if enviar:
    datos = {
        "es_mujer": int(sexo == "Mujer"),
        "edad": int(edad),
        "embarazo": int(embarazo and sexo == "Mujer"),
        "entidad_res": ENTIDADES.index(entidad) + 1,
        **{k: int(v) for k, v in marcadas.items()},
    }
    try:
        entrada = EntradaPaciente(**datos)
    except ValidationError as e:
        st.error("Revisa los datos capturados:")
        for err in e.errors():
            st.write(f"- **{'.'.join(map(str, err['loc'])) or 'registro'}**: {err['msg']}")
        st.stop()

    with st.spinner("Consultando el modelo en Databricks… (si estaba dormido puede tardar hasta 1 min)"):
        inicio = time.perf_counter()
        try:
            pred = cliente.predecir(entrada)
        except ErrorModelo as e:
            st.error(f"**{type(e).__name__}** · {e}")
            st.stop()
        latencia = time.perf_counter() - inicio

    st.subheader(f"{COLOR[pred.nivel_riesgo]} Riesgo {pred.nivel_riesgo.upper()}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidad de hospitalización", f"{pred.probabilidad:.1%}")
    m2.metric("Predicción", "Hospitalización" if pred.hospitalizacion_predicha else "Ambulatorio")
    m3.metric("Latencia", f"{latencia:.2f} s")
    st.progress(min(max(pred.probabilidad, 0.0), 1.0))

    with st.expander("Ver solicitud y respuesta del API"):
        st.code(f"POST {cliente.url_invocacion}", language="text")
        st.code(json.dumps({"dataframe_records": [entrada.model_dump()]}, indent=2), language="json")
        st.code(json.dumps({"predictions": [pred.model_dump()]}, indent=2), language="json")
