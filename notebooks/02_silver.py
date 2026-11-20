# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Capa Silver: limpieza y estandarización
# MAGIC
# MAGIC Se lee **Bronze** (nunca los archivos) y se genera una tabla limpia y tipada. Cada decisión está justificada:
# MAGIC
# MAGIC | Problema | Tratamiento | Justificación |
# MAGIC |---|---|---|
# MAGIC | **Duplicados exactos** | `dropDuplicates` sobre las columnas de negocio | Al cargar varios cortes/años la misma fila puede venir dos veces. |
# MAGIC | **Duplicados por paciente** (`ID_REGISTRO`) | Se conserva el registro con la `FECHA_ACTUALIZACION` más reciente | El mismo caso se actualiza en cortes posteriores (p. ej. pasa de ambulatorio a hospitalizado); la versión más nueva es la correcta. |
# MAGIC | **Códigos de "nulo"** 97 / 98 / 99 | Se convierten a `NULL` | El diccionario de datos los define como *No aplica / Se ignora / No especificado*; dejarlos como número haría creer al modelo que son un valor real. |
# MAGIC | **Codificación SI/NO** 1 / 2 | Se recodifica a 1 / 0 | Convención binaria estándar para modelos. |
# MAGIC | **Embarazo en hombres** (97 = No aplica) | Se imputa 0 | Lógicamente no hay embarazo; no es un dato faltante. |
# MAGIC | **Tipos de dato** | Texto → `int` / `date` con `try_cast` | Bronze guarda todo como texto; `try_cast` evita que una fila mal capturada tumbe el pipeline (queda NULL y se reporta). |
# MAGIC | **Fecha de defunción** `9999-99-99` | → `NULL` + bandera `fallecio` | Es el valor centinela de "no falleció". |
# MAGIC | **Outliers de edad** | Se eliminan edades < 0 o > 110 | Regla de dominio: son errores de captura. No se usa IQR porque recortaría adultos mayores reales, que son justo el grupo de mayor riesgo. |
# MAGIC | **Objetivo nulo** (`TIPO_PACIENTE` no especificado) | Se elimina la fila | Sin etiqueta no sirve para entrenar ni evaluar. |
# MAGIC | **Entidad fuera de catálogo** (97–99) | → `NULL` | Solo 1–32 son claves INEGI válidas. |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

bronze = spark.table(T_BRONZE)
n_inicial = bronze.count()
calidad = [("silver", "entrada_desde_bronze", n_inicial, "Filas leídas de Bronze")]

# --- 1. Duplicados exactos -------------------------------------------------
columnas_negocio = [c for c in bronze.columns if not c.startswith("_")]
df = bronze.dropDuplicates(columnas_negocio)
n_sin_dup_exactos = df.count()
calidad.append(("silver", "duplicados_exactos_eliminados", n_inicial - n_sin_dup_exactos,
                "Filas idénticas en todas las columnas de negocio"))

# --- 2. Duplicados por ID_REGISTRO: conservar la actualización más reciente --
def a_fecha(col):
    """Convierte texto a fecha aceptando yyyy-MM-dd y dd/MM/yyyy; inválidos → NULL."""
    return F.coalesce(
        F.expr(f"try_cast({col} AS DATE)"),
        F.expr(f"CAST(try_to_timestamp({col}, 'dd/MM/yyyy') AS DATE)"),
    )

df = df.withColumn("fecha_actualizacion", a_fecha("FECHA_ACTUALIZACION"))
w = Window.partitionBy("ID_REGISTRO").orderBy(F.col("fecha_actualizacion").desc_nulls_last())
df = df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")
n_sin_dup_id = df.count()
calidad.append(("silver", "duplicados_id_registro_eliminados", n_sin_dup_exactos - n_sin_dup_id,
                "Mismo paciente en varios cortes; se conserva el más reciente"))

# COMMAND ----------

# --- 3. Tipos de dato y recodificación -------------------------------------
def entero(col):
    return F.expr(f"try_cast({col} AS INT)")

def si_no(col):
    """1=SI → 1, 2=NO → 0, 97/98/99 (no aplica / se ignora / no especificado) → NULL."""
    x = entero(col)
    return F.when(x == 1, 1).when(x == 2, 0).otherwise(F.lit(None).cast("int"))

mapa_comorbilidades = {
    "diabetes": "DIABETES",
    "epoc": "EPOC",
    "asma": "ASMA",
    "inmunosupresion": "INMUSUPR",
    "hipertension": "HIPERTENSION",
    "cardiovascular": "CARDIOVASCULAR",
    "obesidad": "OBESIDAD",
    "renal_cronica": "RENAL_CRONICA",
    "tabaquismo": "TABAQUISMO",
    "otra_comorbilidad": "OTRA_COM",
}

sexo = entero("SEXO")
fecha_def_txt = F.trim(F.col("FECHA_DEF"))

silver = df.select(
    F.col("ID_REGISTRO").alias("id_registro"),
    "fecha_actualizacion",
    a_fecha("FECHA_INGRESO").alias("fecha_ingreso"),
    a_fecha("FECHA_SINTOMAS").alias("fecha_sintomas"),
    F.when(fecha_def_txt.startswith("9999"), None).otherwise(a_fecha("FECHA_DEF")).alias("fecha_def"),
    F.when(sexo == 1, 1).when(sexo == 2, 0).otherwise(F.lit(None).cast("int")).alias("es_mujer"),
    entero("EDAD").alias("edad"),
    F.when(entero("ENTIDAD_RES").between(1, 32), entero("ENTIDAD_RES")).alias("entidad_res"),
    F.when(entero("ENTIDAD_UM").between(1, 32), entero("ENTIDAD_UM")).alias("entidad_um"),
    entero("SECTOR").alias("sector"),
    # Embarazo: 97 (no aplica) en hombres → 0; resto con la regla SI/NO
    F.when(sexo == 2, F.lit(0)).otherwise(si_no("EMBARAZO")).alias("embarazo"),
    *[si_no(origen).alias(destino) for destino, origen in mapa_comorbilidades.items()],
    si_no("INDIGENA").alias("indigena"),
    # Variables que ocurren DESPUÉS del ingreso: se guardan para análisis,
    # pero NO se usan como predictores (fuga de información / data leakage).
    si_no("NEUMONIA").alias("neumonia"),
    si_no("INTUBADO").alias("intubado"),
    si_no("UCI").alias("uci"),
    entero("CLASIFICACION_FINAL").alias("clasificacion_final"),
    # Objetivo: 1 = ambulatorio → 0, 2 = hospitalizado → 1
    F.when(entero("TIPO_PACIENTE") == 2, 1)
    .when(entero("TIPO_PACIENTE") == 1, 0)
    .otherwise(F.lit(None).cast("int"))
    .alias("hospitalizado"),
    "_archivo_origen",
).withColumn("fallecio", F.col("fecha_def").isNotNull().cast("int"))

# COMMAND ----------

# --- 4. Reporte de nulos ANTES de filtrar (evidencia para el README) -------
cols_reporte = ["es_mujer", "edad", "entidad_res", "embarazo", "hospitalizado",
                "clasificacion_final", *mapa_comorbilidades.keys()]
nulos = silver.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols_reporte]
).first().asDict()
for c, n in nulos.items():
    calidad.append(("silver", f"nulos_{c}", int(n or 0), "Valores nulos tras convertir 97/98/99 y tipos"))

# --- 5. Outliers de edad y objetivo nulo -----------------------------------
antes = silver.count()
silver = silver.filter(F.col("edad").between(0, 110))
despues_edad = silver.count()
calidad.append(("silver", "outliers_edad_eliminados", antes - despues_edad,
                "Edad nula, < 0 o > 110 (error de captura)"))

silver = silver.filter(F.col("hospitalizado").isNotNull())
n_final = silver.count()
calidad.append(("silver", "objetivo_nulo_eliminado", despues_edad - n_final,
                "TIPO_PACIENTE no especificado"))
calidad.append(("silver", "registros_finales", n_final, "Filas en la tabla Silver"))

(
    silver.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(T_SILVER)
)

spark.createDataFrame(calidad, "capa string, paso string, filas long, descripcion string") \
    .withColumn("_fecha", F.current_timestamp()) \
    .write.mode("append").saveAsTable(T_CALIDAD)

display(spark.table(T_CALIDAD).orderBy("_fecha"))
