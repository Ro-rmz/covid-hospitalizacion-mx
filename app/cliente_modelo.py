"""
Cliente del endpoint publicado en Databricks Model Serving.

La app NUNCA carga el modelo localmente: todas las predicciones se piden por
HTTP al endpoint. Este módulo se encarga de:
  * validar la entrada ANTES de enviarla (Pydantic),
  * timeouts de conexión y de lectura,
  * reintentos con espera exponencial ante fallas temporales
    (timeout, red, 429, 5xx; p. ej. el endpoint "despertando" de scale-to-zero),
  * traducir cada error HTTP a un mensaje entendible,
  * validar que la RESPUESTA tenga el formato esperado.
"""
from __future__ import annotations

import time
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Binaria = Field(ge=0, le=1)


class EntradaPaciente(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    es_mujer: int = Binaria
    edad: int = Field(ge=0, le=110)
    embarazo: int = Binaria
    entidad_res: int = Field(ge=1, le=32)
    diabetes: int = Binaria
    epoc: int = Binaria
    asma: int = Binaria
    inmunosupresion: int = Binaria
    hipertension: int = Binaria
    cardiovascular: int = Binaria
    obesidad: int = Binaria
    renal_cronica: int = Binaria
    tabaquismo: int = Binaria
    otra_comorbilidad: int = Binaria

    @model_validator(mode="after")
    def embarazo_solo_mujeres(self):
        if self.embarazo == 1 and self.es_mujer == 0:
            raise ValueError("embarazo = 1 solo es válido si es_mujer = 1")
        return self


class Prediccion(BaseModel):
    probabilidad: float = Field(ge=0.0, le=1.0)
    hospitalizacion_predicha: Literal[0, 1]
    nivel_riesgo: Literal["bajo", "medio", "alto"]


# ---------------------------------------------------------------- errores
class ErrorModelo(Exception):
    """Base: trae un mensaje listo para mostrar al usuario."""


class ErrorConfiguracion(ErrorModelo): ...
class ErrorAutenticacion(ErrorModelo): ...
class ErrorEntrada(ErrorModelo): ...
class ErrorEndpointNoEncontrado(ErrorModelo): ...
class ErrorServicioNoDisponible(ErrorModelo): ...
class RespuestaInvalida(ErrorModelo): ...


REINTENTABLES = {429, 500, 502, 503, 504}


class ClienteModelo:
    def __init__(
        self,
        host: str,
        token: str,
        endpoint: str,
        timeout_conexion: float = 5.0,
        timeout_lectura: float = 60.0,
        reintentos: int = 3,
        espera_base: float = 2.0,
        session: requests.Session | None = None,
    ):
        if not host or not token or not endpoint:
            raise ErrorConfiguracion(
                "Faltan DATABRICKS_HOST, DATABRICKS_TOKEN o ENDPOINT_NAME en el archivo .env"
            )
        self.host = host.rstrip("/")
        if not self.host.startswith("https://"):
            raise ErrorConfiguracion("DATABRICKS_HOST debe empezar con https://")
        self.endpoint = endpoint
        self.timeout = (timeout_conexion, timeout_lectura)
        self.reintentos = reintentos
        self.espera_base = espera_base
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )

    @property
    def url_invocacion(self) -> str:
        return f"{self.host}/serving-endpoints/{self.endpoint}/invocations"

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _mensaje_error(resp: requests.Response) -> str:
        try:
            cuerpo = resp.json()
            return cuerpo.get("message") or cuerpo.get("error_code") or str(cuerpo)
        except ValueError:
            return (resp.text or "").strip()[:300]

    def _post_con_reintentos(self, payload: dict) -> requests.Response:
        ultimo_error = ""
        for intento in range(1, self.reintentos + 1):
            try:
                resp = self.session.post(self.url_invocacion, json=payload, timeout=self.timeout)
            except requests.Timeout:
                ultimo_error = f"El endpoint no respondió en {self.timeout[1]:.0f} s"
            except requests.ConnectionError:
                ultimo_error = "No se pudo conectar con Databricks (revisa tu internet o el host)"
            else:
                if resp.status_code not in REINTENTABLES:
                    return resp
                ultimo_error = f"HTTP {resp.status_code}: {self._mensaje_error(resp)}"
            if intento < self.reintentos:
                time.sleep(self.espera_base * 2 ** (intento - 1))
        raise ErrorServicioNoDisponible(
            f"El modelo no está disponible después de {self.reintentos} intentos. "
            f"Si el endpoint estaba apagado (scale-to-zero) puede tardar unos minutos en despertar. "
            f"Detalle: {ultimo_error}"
        )

    # ---------------------------------------------------------------- API
    def predecir(self, entrada: EntradaPaciente) -> Prediccion:
        payload = {"dataframe_records": [entrada.model_dump()]}
        resp = self._post_con_reintentos(payload)

        if resp.status_code in (401, 403):
            raise ErrorAutenticacion("Token de Databricks inválido, vencido o sin permiso sobre el endpoint.")
        if resp.status_code == 404:
            raise ErrorEndpointNoEncontrado(f"No existe el endpoint '{self.endpoint}' en {self.host}.")
        if 400 <= resp.status_code < 500:
            raise ErrorEntrada(f"El modelo rechazó los datos: {self._mensaje_error(resp)}")
        if resp.status_code != 200:
            raise ErrorServicioNoDisponible(f"HTTP {resp.status_code}: {self._mensaje_error(resp)}")

        try:
            cuerpo = resp.json()
        except ValueError as e:
            raise RespuestaInvalida("La respuesta del endpoint no es JSON válido.") from e

        predicciones = cuerpo.get("predictions") if isinstance(cuerpo, dict) else None
        if not isinstance(predicciones, list) or len(predicciones) != 1:
            raise RespuestaInvalida(f"Formato de respuesta inesperado: {str(cuerpo)[:200]}")
        try:
            return Prediccion.model_validate(predicciones[0])
        except ValidationError as e:
            raise RespuestaInvalida(f"La predicción no tiene el formato esperado: {e.errors()[:2]}") from e

    def estado(self) -> str:
        """READY / NOT_READY / desconocido (para mostrarlo en la barra lateral)."""
        try:
            resp = self.session.get(
                f"{self.host}/api/2.0/serving-endpoints/{self.endpoint}", timeout=self.timeout
            )
            if resp.status_code != 200:
                return f"desconocido (HTTP {resp.status_code})"
            return resp.json().get("state", {}).get("ready", "desconocido")
        except (requests.RequestException, ValueError):
            return "desconocido (sin conexión)"
