# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Capa Bronze: ingesta cruda
# MAGIC
# MAGIC **Regla de la capa:** los datos se cargan **tal como vienen** en los archivos de la Dirección General de Epidemiología.
# MAGIC No se corrige, no se filtra, no se cambian tipos (todo se guarda como texto). Solo se agregan columnas de **linaje**
# MAGIC (`_archivo_origen`, `_fecha_ingesta`) para saber de dónde vino cada fila. Los archivos originales nunca se modifican.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import glob
import os
import shutil
import zipfile

from pyspark.sql import functions as F

# 1) Descomprimir los .zip que subiste al Volume (solo la primera vez)
os.makedirs(CSV_PATH, exist_ok=True)
zips = sorted(glob.glob(f"{RAW_PATH}/*.zip"))
for ruta_zip in zips:
    with zipfile.ZipFile(ruta_zip) as zf:
        for miembro in zf.namelist():
            if not miembro.lower().endswith(".csv"):
                continue
            destino = os.path.join(CSV_PATH, os.path.basename(miembro))
            if os.path.exists(destino):
                print(f"Ya existe, se omite: {destino}")
                continue
            print(f"Descomprimiendo {miembro} → {destino}")
            with zf.open(miembro) as src, open(destino, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

archivos = sorted(glob.glob(f"{RAW_PATH}/*.csv") + glob.glob(f"{CSV_PATH}/*.csv"))
if not archivos:
    raise FileNotFoundError(
        f"No hay archivos en {RAW_PATH}. Sube ahí los .zip (o .csv) de datos abiertos COVID-19 "
        "de la Secretaría de Salud (Catalog → workspace → covid_mx → raw → Upload to this volume)."
    )
print("Archivos a ingerir:", *archivos, sep="\n  ")

# COMMAND ----------

# 2) Leer cada archivo como texto (sin inferir esquema) y unirlos por nombre de columna.
#    Así, si un año trae una columna extra, no se pierde ni se rompe la carga.
bronze = None
for ruta in archivos:
    df = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("multiLine", False)
        .csv(ruta)
        .withColumn("_archivo_origen", F.col("_metadata.file_name"))
        .withColumn("_fecha_ingesta", F.current_timestamp())
    )
    bronze = df if bronze is None else bronze.unionByName(df, allowMissingColumns=True)

(
    bronze.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(T_BRONZE)
)

# COMMAND ----------

n_bronze = spark.table(T_BRONZE).count()
print(f"Registros en Bronze: {n_bronze:,}")
display(
    spark.table(T_BRONZE).groupBy("_archivo_origen").count().orderBy("_archivo_origen")
)

if n_bronze < MIN_REGISTROS:
    raise ValueError(
        f"Bronze tiene {n_bronze:,} registros; el proyecto pide al menos {MIN_REGISTROS:,}. "
        "Sube otro año / corte de la base."
    )

spark.createDataFrame(
    [("bronze", "registros_crudos", n_bronze, "Filas tal como vienen en los archivos fuente")],
    "capa string, paso string, filas long, descripcion string",
).withColumn("_fecha", F.current_timestamp()).write.mode("overwrite").option(
    "overwriteSchema", True
).saveAsTable(T_CALIDAD)  # cada corrida del pipeline reinicia el reporte de calidad
