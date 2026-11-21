"""
Lógica del modelo de riesgo de hospitalización por COVID-19.

Este módulo NO depende de Spark ni de MLflow para que se pueda:
  * probar en local (pytest),
  * reutilizar en el notebook de entrenamiento,
  * empaquetar junto con el modelo publicado (code_paths de MLflow),
    de modo que el endpoint aplique exactamente la misma validación
    y las mismas transformaciones que se usaron al entrenar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Contrato de entrada (lo que recibe el API) y variable objetivo
# --------------------------------------------------------------------------
COMORBILIDADES = [
    "diabetes",
    "epoc",
    "asma",
    "inmunosupresion",
    "hipertension",
    "cardiovascular",
    "obesidad",
    "renal_cronica",
    "tabaquismo",
    "otra_comorbilidad",
]
BINARIAS = ["es_mujer", "embarazo"] + COMORBILIDADES
FEATURES_ENTRADA = ["edad", "entidad_res"] + BINARIAS
TARGET = "hospitalizado"

EDAD_MIN, EDAD_MAX = 0, 110
ENTIDAD_MIN, ENTIDAD_MAX = 1, 32

# Variables derivadas (se calculan aquí, no las manda el usuario)
CORTES_EDAD = [-1, 17, 39, 59, 74, 200]
ETIQUETAS_EDAD = ["0-17", "18-39", "40-59", "60-74", "75+"]

NUMERICAS = ["edad", "n_comorbilidades"]
CATEGORICAS = ["entidad_res", "grupo_edad"]


class EntradaInvalida(ValueError):
    """Error de validación de entrada (el endpoint lo regresa como 400)."""


def validar_entrada(df: pd.DataFrame) -> None:
    """Valida el lote de registros que llega al modelo. Lanza EntradaInvalida."""
    errores: list[str] = []

    faltantes = [c for c in FEATURES_ENTRADA if c not in df.columns]
    if faltantes:
        raise EntradaInvalida(f"Faltan columnas obligatorias: {faltantes}")
    if len(df) == 0:
        raise EntradaInvalida("La solicitud no contiene registros.")

    sub = df[FEATURES_ENTRADA]
    nulos = sub.columns[sub.isna().any()].tolist()
    if nulos:
        raise EntradaInvalida(f"Hay valores nulos en: {nulos}")

    for c in FEATURES_ENTRADA:
        if not pd.api.types.is_numeric_dtype(sub[c]):
            errores.append(f"'{c}' debe ser numérico (entero)")
    if errores:
        raise EntradaInvalida("; ".join(errores))

    for c in FEATURES_ENTRADA:
        if not np.all(np.equal(np.mod(sub[c].to_numpy(dtype=float), 1), 0)):
            errores.append(f"'{c}' debe ser entero")

    for c in BINARIAS:
        if not sub[c].isin([0, 1]).all():
            errores.append(f"'{c}' solo acepta 0 o 1")

    if not sub["edad"].between(EDAD_MIN, EDAD_MAX).all():
        errores.append(f"'edad' debe estar entre {EDAD_MIN} y {EDAD_MAX}")

    if not sub["entidad_res"].between(ENTIDAD_MIN, ENTIDAD_MAX).all():
        errores.append(
            f"'entidad_res' debe ser una clave de entidad INEGI entre {ENTIDAD_MIN} y {ENTIDAD_MAX}"
        )

    if ((sub["embarazo"] == 1) & (sub["es_mujer"] == 0)).any():
        errores.append("'embarazo' = 1 solo es válido si 'es_mujer' = 1")

    if errores:
        raise EntradaInvalida("; ".join(errores))


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables derivadas usadas por el modelo."""
    out = df[FEATURES_ENTRADA].astype("int64").copy()
    out["n_comorbilidades"] = out[COMORBILIDADES].sum(axis=1)
    out["grupo_edad"] = pd.cut(
        out["edad"], bins=CORTES_EDAD, labels=ETIQUETAS_EDAD
    ).astype(str)
    out["entidad_res"] = out["entidad_res"].astype(str)  # categórica, no ordinal
    return out


def construir_preprocesador() -> ColumnTransformer:
    """Normalización (StandardScaler) y codificación (One-Hot)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
            ("bin", "passthrough", BINARIAS),
        ]
    )


def construir_modelos(seed: int = 42) -> dict[str, Pipeline]:
    """Modelos candidatos. Ambos usan pesos de clase por el desbalance."""
    return {
        "regresion_logistica": Pipeline(
            [
                ("prep", construir_preprocesador()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", construir_preprocesador()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def umbral_optimo_f1(y_true, prob) -> float:
    """Umbral de decisión que maximiza F1 en el set de validación."""
    precision, recall, umbrales = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision_recall_curve regresa len(umbrales) = len(precision) - 1
    idx = int(np.nanargmax(f1[:-1]))
    return float(umbrales[idx])


def metricas(y_true, prob, umbral: float) -> dict[str, float]:
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "f1": f1_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def nivel_riesgo(prob: np.ndarray, umbral: float) -> np.ndarray:
    """alto: >= umbral | medio: >= umbral/2 | bajo: el resto."""
    return np.where(prob >= umbral, "alto", np.where(prob >= umbral / 2, "medio", "bajo"))


@dataclass
class ResultadoEntrenamiento:
    nombre_modelo: str
    modelo: Pipeline
    umbral: float
    metricas_validacion: dict
    metricas_prueba: dict
    comparativa: dict = field(default_factory=dict)


def entrenar_y_evaluar(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    X_test: pd.DataFrame,
    y_test,
    seed: int = 42,
) -> ResultadoEntrenamiento:
    """
    Entrena los candidatos, elige el mejor por F1 en VALIDACIÓN (con su umbral
    óptimo) y reporta las métricas finales en PRUEBA (datos nunca vistos).
    """
    Xtr, Xva, Xte = (preparar_features(X) for X in (X_train, X_val, X_test))
    comparativa, mejor = {}, None

    for nombre, pipe in construir_modelos(seed).items():
        pipe.fit(Xtr, y_train)
        prob_val = pipe.predict_proba(Xva)[:, 1]
        umbral = umbral_optimo_f1(y_val, prob_val)
        m_val = metricas(y_val, prob_val, umbral)
        comparativa[nombre] = {"umbral": umbral, **{f"val_{k}": v for k, v in m_val.items()}}
        if mejor is None or m_val["f1"] > mejor[3]["f1"]:
            mejor = (nombre, pipe, umbral, m_val)

    nombre, pipe, umbral, m_val = mejor
    prob_test = pipe.predict_proba(Xte)[:, 1]
    m_test = metricas(y_test, prob_test, umbral)
    return ResultadoEntrenamiento(nombre, pipe, umbral, m_val, m_test, comparativa)


def f1_linea_base(y_true) -> float:
    """F1 de un modelo trivial que marca a TODOS como hospitalizados (recall = 1)."""
    p = float(np.mean(y_true))
    return 2 * p / (1 + p)


def importancia_permutacion(
    modelo: Pipeline, X: pd.DataFrame, y, n_repeticiones: int = 3, seed: int = 42
) -> pd.Series:
    """
    Importancia por permutación sobre las variables de ENTRADA (las que captura el usuario):
    cuánto cae el PR-AUC si se revuelve esa columna. Las comorbilidades también alimentan
    n_comorbilidades y la edad alimenta grupo_edad, así que el efecto total queda en cada una.
    """
    rng = np.random.default_rng(seed)
    X = X[FEATURES_ENTRADA].reset_index(drop=True)
    base = average_precision_score(y, modelo.predict_proba(preparar_features(X))[:, 1])
    caidas = {}
    for c in FEATURES_ENTRADA:
        valores = []
        for _ in range(n_repeticiones):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            valores.append(base - average_precision_score(y, modelo.predict_proba(preparar_features(Xp))[:, 1]))
        caidas[c] = float(np.mean(valores))
    return pd.Series(caidas).sort_values(ascending=False)


def calibrar(prob, prevalencia: float | None):
    """
    Los modelos se entrenan con class_weight="balanced", que equivale a fingir que
    hay 50 % de hospitalizados. Eso sirve para detectar la clase minoritaria, pero infla
    las probabilidades. Esta corrección de prior regresa cada puntaje a la escala real:
    momios_reales = momios_modelo × (π / (1 − π)), con π = % real de hospitalizados.
    Es monótona: no cambia el orden ni las decisiones, solo la escala.
    """
    prob = np.clip(np.asarray(prob, dtype="float64"), 1e-9, 1 - 1e-9)
    if prevalencia is None:
        return prob
    momios = prob / (1 - prob) * (prevalencia / (1 - prevalencia))
    return momios / (1 + momios)


def predecir(
    modelo: Pipeline, umbral: float, df: pd.DataFrame, prevalencia: float | None = None
) -> pd.DataFrame:
    """Valida, transforma y predice. Es lo que ejecuta el endpoint."""
    validar_entrada(df)
    prob = calibrar(modelo.predict_proba(preparar_features(df))[:, 1], prevalencia)
    umbral_cal = float(calibrar([umbral], prevalencia)[0])
    return pd.DataFrame(
        {
            "probabilidad": prob.astype("float64"),
            "hospitalizacion_predicha": (prob >= umbral_cal).astype("int64"),
            "nivel_riesgo": nivel_riesgo(prob, umbral_cal),
        }
    )
