# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Publicación del modelo como endpoint (Databricks Model Serving)
# MAGIC
# MAGIC Crea (o actualiza) el endpoint REST `covid-hospitalizacion` con la versión **champion** del modelo.
# MAGIC - CPU, tamaño *Small*, **scale-to-zero** (no consume si nadie lo usa; la primera llamada tras estar dormido tarda más,
# MAGIC   y la app lo maneja con timeouts y reintentos).
# MAGIC - La **validación de entradas** ocurre en dos niveles dentro del endpoint: (1) la *signature* de MLflow rechaza columnas o tipos
# MAGIC   incorrectos y (2) `modelo_covid.validar_entrada` revisa rangos y reglas de negocio (edad 0–110, binarias 0/1, entidad 1–32,
# MAGIC   embarazo solo en mujeres). Si algo falla, responde **400** con el motivo.
# MAGIC - Al final se descarga la especificación **OpenAPI** que genera Databricks para el endpoint.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.tracking import MlflowClient

w = WorkspaceClient()
version = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version

entidad = ServedEntityInput(
    entity_name=MODEL_NAME,
    entity_version=str(version),
    workload_size="Small",
    scale_to_zero_enabled=True,
)

existentes = {e.name for e in w.serving_endpoints.list()}
if ENDPOINT_NAME not in existentes:
    print(f"Creando endpoint {ENDPOINT_NAME} con la versión {version} (tarda ~10-20 min)...")
    # Según la versión del SDK, EndpointCoreConfigInput pide (o no acepta) el nombre
    try:
        config = EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[entidad])
    except TypeError:
        config = EndpointCoreConfigInput(served_entities=[entidad])
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=config,
        timeout=timedelta(minutes=45),
    )
else:
    actual = w.serving_endpoints.get(ENDPOINT_NAME)
    versiones_actuales = {e.entity_version for e in (actual.config.served_entities or [])} if actual.config else set()
    if str(version) in versiones_actuales:
        print(f"El endpoint ya sirve la versión {version}; no hay nada que actualizar.")
    else:
        print(f"Actualizando endpoint {ENDPOINT_NAME} a la versión {version}...")
        w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME, served_entities=[entidad], timeout=timedelta(minutes=45)
        )

estado = w.serving_endpoints.get(ENDPOINT_NAME).state
print("Estado:", estado.ready, "| actualización:", estado.config_update)

# COMMAND ----------

# Prueba del endpoint publicado (mismo contrato que usa la app)
ejemplo = {
    "es_mujer": 0, "edad": 67, "embarazo": 0, "entidad_res": 14,
    "diabetes": 1, "epoc": 0, "asma": 0, "inmunosupresion": 0, "hipertension": 1,
    "cardiovascular": 0, "obesidad": 1, "renal_cronica": 0, "tabaquismo": 0, "otra_comorbilidad": 0,
}
respuesta = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=[ejemplo])
print("Predicción:", respuesta.predictions)

# Prueba de validación: edad inválida → debe fallar con 400
try:
    w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=[{**ejemplo, "edad": 150}])
    print("⚠️ El endpoint aceptó una edad inválida")
except Exception as e:  # noqa: BLE001
    print("✅ Entrada inválida rechazada:", str(e)[:300])

# COMMAND ----------

# Especificación OpenAPI generada por Databricks (se guarda en el Volume para el README)
try:
    spec = w.serving_endpoints.get_open_api(ENDPOINT_NAME)
    contenido = spec.contents.read() if hasattr(spec, "contents") else str(spec).encode()
    ruta = f"{RAW_PATH}/openapi_{ENDPOINT_NAME}.json"
    with open(ruta, "wb") as f:
        f.write(contenido)
    print(f"OpenAPI guardado en {ruta}")
    print(contenido[:1500].decode(errors="ignore"))
except Exception as e:  # noqa: BLE001
    print("No se pudo descargar el OpenAPI desde Databricks; usa api/openapi.yaml del repo.", e)

host = w.config.host.rstrip("/")
print(f"\nURL de invocación para la app:\n{host}/serving-endpoints/{ENDPOINT_NAME}/invocations")
