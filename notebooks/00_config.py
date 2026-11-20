# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Configuración compartida
# MAGIC Los demás notebooks lo cargan con `%run ./00_config`. Si cambias el catálogo o el esquema, cámbialo solo aquí.

# COMMAND ----------

CATALOG = "workspace"          # catálogo por defecto en Databricks Free Edition
SCHEMA = "covid_mx"
VOLUME = "raw"

RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"   # aquí subes los .zip / .csv de la Secretaría de Salud
CSV_PATH = f"{RAW_PATH}/csv"                         # aquí se descomprimen

# Tablas de la arquitectura Medallion
T_BRONZE = f"{CATALOG}.{SCHEMA}.bronze_covid"
T_SILVER = f"{CATALOG}.{SCHEMA}.silver_covid"
T_GOLD = f"{CATALOG}.{SCHEMA}.gold_features_hospitalizacion"
T_GOLD_ENTIDAD = f"{CATALOG}.{SCHEMA}.gold_resumen_entidad"
T_GOLD_PERFIL = f"{CATALOG}.{SCHEMA}.gold_resumen_perfil"
T_CALIDAD = f"{CATALOG}.{SCHEMA}.calidad_datos"

# Modelo y endpoint
MODEL_NAME = f"{CATALOG}.{SCHEMA}.modelo_hospitalizacion_covid"
MODEL_ALIAS = "champion"
ENDPOINT_NAME = "covid-hospitalizacion"

# Reglas
MIN_REGISTROS = 4_000_000            # requisito del proyecto
MAX_FILAS_ENTRENAMIENTO = 1_500_000  # muestra estratificada para entrenar (ver README)
SEED = 42

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# COMMAND ----------

import os
import sys

# Permite importar src/modelo_covid.py desde los notebooks (el repo está en un Git folder)
RUTA_SRC = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if RUTA_SRC not in sys.path:
    sys.path.append(RUTA_SRC)

print(f"Catálogo/esquema: {CATALOG}.{SCHEMA} | Volume: {RAW_PATH} | src: {RUTA_SRC}")
