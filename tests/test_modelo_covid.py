import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modelo_covid import (  # noqa: E402
    COMORBILIDADES,
    FEATURES_ENTRADA,
    EntradaInvalida,
    entrenar_y_evaluar,
    predecir,
    validar_entrada,
)


def datos_sinteticos(n=20_000, seed=0):
    """Datos con la misma estructura que Gold; riesgo creciente con edad y comorbilidades."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "edad": rng.integers(0, 100, n),
        "entidad_res": rng.integers(1, 33, n),
        "es_mujer": rng.integers(0, 2, n),
    })
    df["embarazo"] = np.where(df.es_mujer == 1, rng.binomial(1, 0.03, n), 0)
    for c in COMORBILIDADES:
        df[c] = rng.binomial(1, 0.1, n)
    logit = -5 + 0.05 * df.edad + 0.6 * df[COMORBILIDADES].sum(axis=1) - 0.3 * df.es_mujer
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df[FEATURES_ENTRADA], pd.Series(y, name="hospitalizado")


def registro(**cambios):
    base = {c: 0 for c in FEATURES_ENTRADA}
    base.update(edad=50, entidad_res=14)
    base.update(cambios)
    return pd.DataFrame([base])


def test_entrenamiento_y_prediccion():
    X, y = datos_sinteticos()
    n = len(X)
    tr, va = int(n * 0.7), int(n * 0.85)
    res = entrenar_y_evaluar(X[:tr], y[:tr], X[tr:va], y[tr:va], X[va:], y[va:])
    assert set(res.comparativa) == {"regresion_logistica", "gradient_boosting"}
    assert 0 < res.umbral < 1
    assert res.metricas_prueba["roc_auc"] > 0.7
    assert res.metricas_prueba["f1"] > 0.3

    out = predecir(res.modelo, res.umbral, X[va:va + 10].reset_index(drop=True))
    assert list(out.columns) == ["probabilidad", "hospitalizacion_predicha", "nivel_riesgo"]
    assert out.probabilidad.between(0, 1).all()
    assert set(out.nivel_riesgo) <= {"bajo", "medio", "alto"}

    joven = predecir(res.modelo, res.umbral, registro(edad=20)).probabilidad[0]
    mayor = predecir(res.modelo, res.umbral, registro(edad=85, diabetes=1, hipertension=1)).probabilidad[0]
    assert mayor > joven


@pytest.mark.parametrize("cambio, mensaje", [
    ({"edad": 150}, "edad"),
    ({"edad": -1}, "edad"),
    ({"entidad_res": 40}, "entidad_res"),
    ({"diabetes": 2}, "diabetes"),
    ({"embarazo": 1, "es_mujer": 0}, "embarazo"),
])
def test_validacion_rechaza(cambio, mensaje):
    with pytest.raises(EntradaInvalida, match=mensaje):
        validar_entrada(registro(**cambio))


def test_validacion_columnas_y_nulos():
    with pytest.raises(EntradaInvalida, match="Faltan"):
        validar_entrada(registro().drop(columns=["edad"]))
    df = registro().astype(float)
    df.loc[0, "obesidad"] = np.nan
    with pytest.raises(EntradaInvalida, match="nulos"):
        validar_entrada(df)
    with pytest.raises(EntradaInvalida, match="entero"):
        validar_entrada(registro(edad=40.5))


def test_validacion_acepta_registro_valido():
    validar_entrada(registro(es_mujer=1, embarazo=1, diabetes=1))


def test_calibracion_regresa_a_escala_real():
    from modelo_covid import calibrar
    # Con prevalencia 0.5 (lo que "ve" class_weight balanced) no cambia nada
    assert np.allclose(calibrar([0.2, 0.7], 0.5), [0.2, 0.7])
    # Con prevalencia baja, baja la probabilidad pero conserva el orden
    p = calibrar([0.3, 0.6, 0.9], 0.15)
    assert np.all(np.diff(p) > 0) and p[-1] < 0.9
    assert np.allclose(calibrar([0.4], None), [0.4])
