# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Entrenamiento, evaluación y registro del modelo (MLflow + Unity Catalog)
# MAGIC
# MAGIC **Tipo de tarea:** clasificación binaria → ¿el paciente confirmado con COVID-19 será **hospitalizado** (1) o ambulatorio (0)?
# MAGIC
# MAGIC **Métrica principal: F1 de la clase "hospitalizado".** Las clases están desbalanceadas (la mayoría de los casos son ambulatorios),
# MAGIC así que el *accuracy* engaña: un modelo que diga siempre "ambulatorio" tendría accuracy alto y no serviría para nada.
# MAGIC F1 equilibra **recall** (no dejar pasar pacientes que sí se hospitalizan, lo más costoso en salud) y **precision** (no alarmar de más).
# MAGIC Como apoyo se reportan recall, precision, ROC-AUC y PR-AUC.
# MAGIC
# MAGIC **Estrategia:**
# MAGIC - Muestra **estratificada** de Gold (misma proporción de clases) para entrenar con scikit-learn. Spark hace el trabajo pesado
# MAGIC   (limpieza de millones de filas); para un modelo tabular de 13 variables, ~1.5 M de filas son más que suficientes y caben en memoria.
# MAGIC - Partición 70 / 15 / 15 (entrenamiento / validación / prueba), estratificada.
# MAGIC - Dos candidatos con `class_weight="balanced"`: regresión logística (base interpretable) y Gradient Boosting.
# MAGIC - El **umbral de decisión** se ajusta en validación para maximizar F1; la métrica final se reporta en **prueba** (datos nunca vistos).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
import os
import tempfile

import joblib
import mlflow
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from modelo_covid import FEATURES_ENTRADA, TARGET, entrenar_y_evaluar, predecir
from pyfunc_wrapper import ModeloHospitalizacionCovid

mlflow.set_registry_uri("databricks-uc")
usuario = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{usuario}/covid_hospitalizacion_mx")

# COMMAND ----------

gold = spark.table(T_GOLD)
total = gold.count()
fraccion = min(1.0, MAX_FILAS_ENTRENAMIENTO / total)
muestra = gold.sampleBy(TARGET, fractions={0: fraccion, 1: fraccion}, seed=SEED)
pdf = muestra.select(*FEATURES_ENTRADA, TARGET).toPandas()
print(f"Gold: {total:,} filas | muestra estratificada: {len(pdf):,} | % hospitalizados: {pdf[TARGET].mean():.2%}")

X, y = pdf[FEATURES_ENTRADA], pdf[TARGET].astype(int)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

# COMMAND ----------

with mlflow.start_run(run_name="covid_hospitalizacion") as run:
    res = entrenar_y_evaluar(X_train, y_train, X_val, y_val, X_test, y_test, seed=SEED)

    mlflow.log_params({
        "modelo_elegido": res.nombre_modelo,
        "umbral_decision": round(res.umbral, 4),
        "filas_gold": total,
        "filas_entrenamiento": len(X_train),
        "filas_validacion": len(X_val),
        "filas_prueba": len(X_test),
        "proporcion_positivos": round(float(y.mean()), 4),
        "metrica_principal": "f1",
    })
    mlflow.log_metrics({f"test_{k}": v for k, v in res.metricas_prueba.items()})
    for nombre, valores in res.comparativa.items():
        mlflow.log_metrics({f"{nombre}__{k}": v for k, v in valores.items()})

    with tempfile.TemporaryDirectory() as tmp:
        ruta_pipeline = os.path.join(tmp, "pipeline.joblib")
        ruta_config = os.path.join(tmp, "config.json")
        joblib.dump(res.modelo, ruta_pipeline)
        with open(ruta_config, "w") as f:
            json.dump({"umbral": res.umbral, "modelo": res.nombre_modelo}, f)

        ejemplo = X_test.head(5).astype("int64").reset_index(drop=True)
        salida = predecir(res.modelo, res.umbral, ejemplo)
        firma = infer_signature(ejemplo, salida)

        info = mlflow.pyfunc.log_model(
            artifact_path="modelo",
            python_model=ModeloHospitalizacionCovid(),
            artifacts={"pipeline": ruta_pipeline, "config": ruta_config},
            code_paths=[os.path.join(RUTA_SRC, "modelo_covid.py"), os.path.join(RUTA_SRC, "pyfunc_wrapper.py")],
            signature=firma,
            input_example=ejemplo.head(2),
            pip_requirements=[
                f"scikit-learn=={sklearn.__version__}",
                f"pandas=={pd.__version__}",
                f"joblib=={joblib.__version__}",
            ],
            registered_model_name=MODEL_NAME,
        )

print("Modelo elegido:", res.nombre_modelo, "| umbral:", round(res.umbral, 4))
display(pd.DataFrame(res.comparativa).T)
print("Métricas en PRUEBA:", json.dumps({k: round(v, 4) for k, v in res.metricas_prueba.items()}, indent=2))

# COMMAND ----------

# Alias "champion" → la versión que se publica en el endpoint
cliente = MlflowClient(registry_uri="databricks-uc")
version = getattr(info, "registered_model_version", None)
if version is None:
    version = max(int(v.version) for v in cliente.search_model_versions(f"name='{MODEL_NAME}'"))
cliente.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, str(version))
print(f"{MODEL_NAME} versión {version} → alias '{MODEL_ALIAS}'")

# Prueba rápida: cargar el modelo registrado y predecir
cargado = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
display(pd.concat([ejemplo, cargado.predict(ejemplo)], axis=1))
