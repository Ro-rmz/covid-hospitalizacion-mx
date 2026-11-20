import os
import sys
from unittest.mock import MagicMock

import pytest
import requests
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from cliente_modelo import (  # noqa: E402
    ClienteModelo,
    EntradaPaciente,
    ErrorAutenticacion,
    ErrorConfiguracion,
    ErrorEndpointNoEncontrado,
    ErrorEntrada,
    ErrorServicioNoDisponible,
    RespuestaInvalida,
)

ENTRADA = dict(
    es_mujer=0, edad=67, embarazo=0, entidad_res=14, diabetes=1, epoc=0, asma=0,
    inmunosupresion=0, hipertension=1, cardiovascular=0, obesidad=1, renal_cronica=0,
    tabaquismo=0, otra_comorbilidad=0,
)
OK = {"predictions": [{"probabilidad": 0.62, "hospitalizacion_predicha": 1, "nivel_riesgo": "alto"}]}


def respuesta(status, cuerpo=None, texto=""):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    if cuerpo is None:
        r.json.side_effect = ValueError("no json")
    else:
        r.json.return_value = cuerpo
    r.text = texto
    return r


def cliente(*efectos):
    s = MagicMock()
    s.headers = {}
    s.post.side_effect = list(efectos)
    return ClienteModelo("https://dbc-x.cloud.databricks.com", "tok", "covid-hospitalizacion",
                         reintentos=3, espera_base=0, session=s), s


def test_prediccion_ok_y_payload():
    c, s = cliente(respuesta(200, OK))
    pred = c.predecir(EntradaPaciente(**ENTRADA))
    assert pred.probabilidad == 0.62 and pred.nivel_riesgo == "alto"
    args, kwargs = s.post.call_args
    assert args[0].endswith("/serving-endpoints/covid-hospitalizacion/invocations")
    assert kwargs["json"] == {"dataframe_records": [ENTRADA]}
    assert kwargs["timeout"] == (5.0, 60.0)


def test_reintenta_timeout_y_503_luego_ok():
    c, s = cliente(requests.Timeout(), respuesta(503, {"message": "scaling from zero"}), respuesta(200, OK))
    assert c.predecir(EntradaPaciente(**ENTRADA)).hospitalizacion_predicha == 1
    assert s.post.call_count == 3


def test_agota_reintentos():
    c, s = cliente(requests.ConnectionError(), requests.Timeout(), respuesta(504, None, "gateway"))
    with pytest.raises(ErrorServicioNoDisponible, match="3 intentos"):
        c.predecir(EntradaPaciente(**ENTRADA))


@pytest.mark.parametrize("status, error", [
    (401, ErrorAutenticacion), (403, ErrorAutenticacion),
    (404, ErrorEndpointNoEncontrado), (400, ErrorEntrada),
])
def test_errores_http_no_reintentables(status, error):
    c, s = cliente(respuesta(status, {"message": "x"}))
    with pytest.raises(error):
        c.predecir(EntradaPaciente(**ENTRADA))
    assert s.post.call_count == 1


@pytest.mark.parametrize("cuerpo", [
    None,                                    # no es JSON
    {"foo": 1},                              # sin "predictions"
    {"predictions": []},                     # vacío
    {"predictions": [{"probabilidad": 1.7, "hospitalizacion_predicha": 1, "nivel_riesgo": "alto"}]},
    {"predictions": [{"probabilidad": 0.2, "hospitalizacion_predicha": 1, "nivel_riesgo": "rojo"}]},
])
def test_respuestas_invalidas(cuerpo):
    c, _ = cliente(respuesta(200, cuerpo))
    with pytest.raises(RespuestaInvalida):
        c.predecir(EntradaPaciente(**ENTRADA))


@pytest.mark.parametrize("cambio", [
    {"edad": 111}, {"entidad_res": 0}, {"diabetes": 3},
    {"embarazo": 1, "es_mujer": 0}, {"edad": "40"}, {"color": "azul"},
])
def test_validacion_local(cambio):
    with pytest.raises(ValidationError):
        EntradaPaciente(**{**ENTRADA, **cambio})


def test_configuracion_incompleta():
    with pytest.raises(ErrorConfiguracion):
        ClienteModelo("", "tok", "ep")
    with pytest.raises(ErrorConfiguracion):
        ClienteModelo("http://inseguro", "tok", "ep")
