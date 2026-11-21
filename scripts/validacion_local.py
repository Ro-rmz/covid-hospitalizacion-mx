"""
Réplica local (pandas) de la lógica Bronze → Silver → Gold → Entrenamiento.

Sirve para validar el código y los números ANTES de correr el Job en Databricks.
No sustituye al pipeline de PySpark: usa exactamente las mismas reglas y el
mismo módulo src/modelo_covid.py para entrenar.

Uso:  python scripts/validacion_local.py COVID19MEXICO2020.zip COVID19MEXICO2021.zip
"""
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from modelo_covid import (  # noqa: E402
    COMORBILIDADES, FEATURES_ENTRADA, TARGET, entrenar_y_evaluar, f1_linea_base, importancia_permutacion,
)

MAX_FILAS_ENTRENAMIENTO = 1_500_000
SEED = 42
COLS = ["FECHA_ACTUALIZACION", "ID_REGISTRO", "SEXO", "EDAD", "ENTIDAD_RES", "TIPO_PACIENTE",
        "CLASIFICACION_FINAL", "EMBARAZO", "DIABETES", "EPOC", "ASMA", "INMUSUPR", "HIPERTENSION",
        "CARDIOVASCULAR", "OBESIDAD", "RENAL_CRONICA", "TABAQUISMO", "OTRA_COM"]
MAPA = {"diabetes": "DIABETES", "epoc": "EPOC", "asma": "ASMA", "inmunosupresion": "INMUSUPR",
        "hipertension": "HIPERTENSION", "cardiovascular": "CARDIOVASCULAR", "obesidad": "OBESIDAD",
        "renal_cronica": "RENAL_CRONICA", "tabaquismo": "TABAQUISMO", "otra_comorbilidad": "OTRA_COM"}


def leer(zips):
    """Lee por bloques para caber en memoria; guarda un hash de la fila COMPLETA
    (todas las columnas de negocio) para detectar duplicados exactos como en Silver."""
    partes = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            nombre = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            with zf.open(nombre) as fh:
                for bloque in pd.read_csv(fh, dtype=str, encoding="latin-1", chunksize=250_000):
                    h = pd.util.hash_pandas_object(bloque, index=False).to_numpy()
                    b = bloque[COLS].copy()
                    for c in COLS[2:]:
                        b[c] = pd.to_numeric(b[c], errors="coerce").astype("float32")
                    b["FECHA_ACTUALIZACION"] = pd.to_numeric(
                        b["FECHA_ACTUALIZACION"].str.replace("-", "", regex=False), errors="coerce"
                    ).astype("float64")
                    b["ID_REGISTRO"] = pd.util.hash_array(b["ID_REGISTRO"].to_numpy(dtype=object))  # hash para ahorrar memoria
                    b["_hash_fila"] = h
                    b["_archivo_origen"] = np.int8(zips.index(z))
                    partes.append(b)
    return pd.concat(partes, ignore_index=True)


def si_no(s):
    x = pd.to_numeric(s, errors="coerce")
    return pd.Series(np.where(x == 1, 1, np.where(x == 2, 0, np.nan)), index=s.index)


def main(zips):
    t0 = time.time()
    calidad = []
    bronze = leer(zips)
    calidad.append(("bronze", "registros_crudos", len(bronze)))

    # --- Silver ---
    df = bronze.drop_duplicates(subset=["_hash_fila"])
    del bronze
    calidad.append(("silver", "duplicados_exactos_eliminados", calidad[0][2] - len(df)))
    n1 = len(df)
    df = df.sort_values("FECHA_ACTUALIZACION", ascending=False).drop_duplicates("ID_REGISTRO")
    calidad.append(("silver", "duplicados_id_registro_eliminados", n1 - len(df)))

    sexo = pd.to_numeric(df["SEXO"], errors="coerce")
    s = pd.DataFrame(index=df.index)
    s["es_mujer"] = np.where(sexo == 1, 1, np.where(sexo == 2, 0, np.nan))
    s["edad"] = pd.to_numeric(df["EDAD"], errors="coerce")
    ent = pd.to_numeric(df["ENTIDAD_RES"], errors="coerce")
    s["entidad_res"] = ent.where(ent.between(1, 32))
    s["embarazo"] = np.where(sexo == 2, 0, si_no(df["EMBARAZO"]))
    for destino, origen in MAPA.items():
        s[destino] = si_no(df[origen])
    s["clasificacion_final"] = pd.to_numeric(df["CLASIFICACION_FINAL"], errors="coerce")
    tp = pd.to_numeric(df["TIPO_PACIENTE"], errors="coerce")
    s[TARGET] = np.where(tp == 2, 1, np.where(tp == 1, 0, np.nan))

    for c in ["es_mujer", "edad", "entidad_res", "embarazo", TARGET, *MAPA]:
        calidad.append(("silver", f"nulos_{c}", int(s[c].isna().sum())))
    antes = len(s)
    s = s[s["edad"].between(0, 110)]
    calidad.append(("silver", "outliers_edad_eliminados", antes - len(s)))
    antes = len(s)
    s = s[s[TARGET].notna()]
    calidad.append(("silver", "objetivo_nulo_eliminado", antes - len(s)))
    calidad.append(("silver", "registros_finales", len(s)))

    # --- Gold ---
    conf = s[s["clasificacion_final"].isin([1, 2, 3])]
    gold = conf.dropna(subset=FEATURES_ENTRADA)[FEATURES_ENTRADA + [TARGET]].astype("int64")
    pos = int(gold[TARGET].sum())
    n_conf = len(conf)
    del df, s, conf  # liberar memoria antes de entrenar
    import gc
    gc.collect()
    calidad += [("gold", "casos_confirmados", n_conf),
                ("gold", "eliminados_por_variables_nulas", n_conf - len(gold)),
                ("gold", "registros_finales", len(gold)),
                ("gold", "hospitalizados", pos)]

    gold["n_comorbilidades"] = gold[COMORBILIDADES].sum(axis=1)
    gold["grupo_edad"] = pd.cut(gold["edad"], [-1, 17, 39, 59, 74, 200],
                                labels=["0-17", "18-39", "40-59", "60-74", "75+"])
    perfil = (gold.assign(n_com=gold["n_comorbilidades"].clip(upper=3))
              .groupby(["grupo_edad", "n_com"], observed=True)[TARGET].agg(["count", "mean"]).reset_index())

    # --- Entrenamiento (igual que el notebook 04) ---
    frac = min(1.0, MAX_FILAS_ENTRENAMIENTO / len(gold))
    muestra = gold.groupby(TARGET, group_keys=False).sample(frac=frac, random_state=SEED)
    X, y = muestra[FEATURES_ENTRADA], muestra[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    res = entrenar_y_evaluar(X_tr, y_tr, X_va, y_va, X_te, y_te, seed=SEED)

    imp = importancia_permutacion(res.modelo, X_te.head(200_000), y_te.head(200_000))

    salida = {
        "f1_linea_base": f1_linea_base(y_te),
        "importancia_pr_auc": imp.round(4).to_dict(),
        "calidad": [{"capa": a, "paso": b, "filas": int(c)} for a, b, c in calidad],
        "pct_hospitalizados_gold": pos / len(gold),
        "filas_muestra": len(muestra),
        "modelo_elegido": res.nombre_modelo,
        "umbral": res.umbral,
        "comparativa_validacion": res.comparativa,
        "metricas_prueba": res.metricas_prueba,
        "perfil": perfil.astype({"grupo_edad": str}).to_dict(orient="records"),
        "segundos": round(time.time() - t0, 1),
    }
    print(json.dumps(salida, indent=2, ensure_ascii=False, default=float))
    return salida


if __name__ == "__main__":
    main(sys.argv[1:])
