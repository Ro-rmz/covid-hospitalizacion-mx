"""
Envoltura MLflow (pyfunc) del modelo.

Se publica junto con modelo_covid.py (code_paths) para que el endpoint de
Databricks valide la entrada y calcule las variables derivadas igual que
en el entrenamiento.
"""
import json

import joblib
import mlflow.pyfunc

from modelo_covid import predecir


class ModeloHospitalizacionCovid(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.modelo = joblib.load(context.artifacts["pipeline"])
        with open(context.artifacts["config"]) as f:
            config = json.load(f)
        self.umbral = float(config["umbral"])
        self.prevalencia = config.get("prevalencia")  # para calibrar la probabilidad

    def predict(self, context, model_input, params=None):
        # Si la entrada no es válida, se lanza EntradaInvalida (ValueError)
        # y Databricks Model Serving responde con un error 400.
        return predecir(self.modelo, self.umbral, model_input, self.prevalencia)
