# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Capa Gold: datos listos para el modelo y para análisis
# MAGIC
# MAGIC 1. **`gold_features_hospitalizacion`**: un registro por caso **confirmado** de COVID-19 con solo las variables que se conocen
# MAGIC    *al momento de llegar a consulta* (sexo, edad, embarazo, comorbilidades, entidad) y el objetivo `hospitalizado`.
# MAGIC 2. **`gold_resumen_entidad`** y **`gold_resumen_perfil`**: agregados para el análisis exploratorio o un dashboard.
# MAGIC
# MAGIC **Decisiones:**
# MAGIC - Se filtran casos confirmados (`clasificacion_final` 1, 2 o 3). Un caso negativo no tiene COVID, así que no aporta a la pregunta
# MAGIC   "*si me da COVID, ¿qué tan probable es que me hospitalicen?*".
# MAGIC - Se **excluyen** `neumonia`, `intubado`, `uci` y `fecha_def` como predictores: ocurren *después* o *durante* la hospitalización
# MAGIC   (fuga de información). Un modelo que las use se ve muy bien en la métrica pero es inútil en la vida real.
# MAGIC - Filas con alguna comorbilidad "se ignora" (NULL) se eliminan en vez de imputar "NO": imputar sesgaría hacia pacientes sanos,
# MAGIC   y con millones de registros perder un porcentaje pequeño no afecta. El porcentaje exacto queda en `calidad_datos`.
# MAGIC - La **normalización** (StandardScaler de la edad y del número de comorbilidades) y la **codificación One-Hot** (entidad y grupo de
# MAGIC   edad) se hacen dentro del pipeline de scikit-learn en el notebook 04, para que el endpoint aplique exactamente la misma
# MAGIC   transformación. En Gold ya queda todo codificado como 0/1 y entero.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import functions as F

from modelo_covid import COMORBILIDADES, FEATURES_ENTRADA, TARGET

silver = spark.table(T_SILVER)
confirmados = silver.filter(F.col("clasificacion_final").isin(1, 2, 3))
n_conf = confirmados.count()

gold = confirmados.select("id_registro", *FEATURES_ENTRADA, TARGET).dropna(subset=FEATURES_ENTRADA)
gold = gold.withColumn("n_comorbilidades", sum(F.col(c) for c in COMORBILIDADES))
n_gold = gold.count()

(
    gold.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(T_GOLD)
)

pos = gold.filter(F.col(TARGET) == 1).count()
calidad = [
    ("gold", "casos_confirmados", n_conf, "clasificacion_final en 1, 2, 3"),
    ("gold", "eliminados_por_variables_nulas", n_conf - n_gold,
     "Comorbilidad/sexo/entidad 'se ignora' o 'no especificado'"),
    ("gold", "registros_finales", n_gold, "Filas listas para entrenar"),
    ("gold", "hospitalizados", pos, f"Clase positiva = {pos / n_gold:.2%} → clases desbalanceadas"),
]
# Si este notebook se vuelve a correr, se reemplazan sus filas en vez de duplicarlas
if spark.catalog.tableExists(T_CALIDAD):
    spark.sql(f"DELETE FROM {T_CALIDAD} WHERE capa = 'gold'")

spark.createDataFrame(calidad, "capa string, paso string, filas long, descripcion string") \
    .withColumn("_fecha", F.current_timestamp()) \
    .write.mode("append").saveAsTable(T_CALIDAD)

print(f"Confirmados: {n_conf:,} | Gold: {n_gold:,} | Hospitalizados: {pos:,} ({pos / n_gold:.2%})")

# COMMAND ----------

# Agregados para análisis / dashboard
nombres_entidad = spark.createDataFrame(
    list(enumerate([
        "Aguascalientes", "Baja California", "Baja California Sur", "Campeche", "Coahuila", "Colima",
        "Chiapas", "Chihuahua", "Ciudad de México", "Durango", "Guanajuato", "Guerrero", "Hidalgo",
        "Jalisco", "Estado de México", "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca",
        "Puebla", "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora", "Tabasco",
        "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas",
    ], start=1)),
    "entidad_res int, entidad string",
)

(
    gold.groupBy("entidad_res")
    .agg(F.count("*").alias("casos"), F.avg(TARGET).alias("tasa_hospitalizacion"))
    .join(nombres_entidad, "entidad_res")
    .orderBy(F.desc("tasa_hospitalizacion"))
    .write.mode("overwrite").option("overwriteSchema", True).saveAsTable(T_GOLD_ENTIDAD)
)

grupo_edad = (
    F.when(F.col("edad") <= 17, "0-17").when(F.col("edad") <= 39, "18-39")
    .when(F.col("edad") <= 59, "40-59").when(F.col("edad") <= 74, "60-74").otherwise("75+")
)
(
    gold.withColumn("grupo_edad", grupo_edad)
    .withColumn("n_comorbilidades_cat", F.least(F.col("n_comorbilidades"), F.lit(3)))
    .groupBy("grupo_edad", "n_comorbilidades_cat")
    .agg(F.count("*").alias("casos"), F.avg(TARGET).alias("tasa_hospitalizacion"))
    .orderBy("grupo_edad", "n_comorbilidades_cat")
    .write.mode("overwrite").option("overwriteSchema", True).saveAsTable(T_GOLD_PERFIL)
)

display(spark.table(T_GOLD_PERFIL))
