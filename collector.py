

"""
Pachuca Mobility — Traffic Collector v1.0

Automated TomTom traffic collector for the Boulevard Felipe Ángeles corridor.

This script:
1. Queries TomTom for the six validated monitoring points.
2. Builds fact_traffic_observation.
3. Validates the capture.
4. Persists the capture into the monthly CSV without duplicating observation_id.

API key:
    Set the TOMTOM_API_KEY environment variable.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import requests


TOMTOM_URL = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
)

CORRIDOR_ID = "COR001"

ARCHIVO_TRAFFIC = os.path.join(
    "data", "silver", "traffic", "2026-09.csv"
)

PUNTOS_MONITOREO = {
    "MP001": {
        "name": "Unidad Deportiva",
        "lat": 20.069184,
        "lon": -98.779956,
        "segment_id": "S001",
    },
    "MP002": {
        "name": "SAT",
        "lat": 20.062865,
        "lon": -98.783239,
        "segment_id": "S001",
    },
    "MP003": {
        "name": "ADO Villas",
        "lat": 20.055266,
        "lon": -98.787273,
        "segment_id": "S002",
    },
    "MP004": {
        "name": "Puente a Av. las Torres",
        "lat": 20.041287,
        "lon": -98.794815,
        "segment_id": "S002",
    },
    "MP005": {
        "name": "Puente Matilde",
        "lat": 20.030988,
        "lon": -98.802909,
        "segment_id": "S003",
    },
    "MP006": {
        "name": "Salida Pachuca-CDMX",
        "lat": 20.028148,
        "lon": -98.805393,
        "segment_id": "S003",
    },
}


def consultar_segmento(lat, lon, api_key):
    """Consulta TomTom Flow Segment Data para un punto."""
    params = {
        "key": api_key,
        "point": f"{lat},{lon}",
        "unit": "kmph",
        "openLr": "true",
    }

    response = requests.get(
        TOMTOM_URL,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"TomTom HTTP {response.status_code}: {response.text}"
        )

    flow = response.json()["flowSegmentData"]

    current_speed = flow["currentSpeed"]
    free_flow_speed = flow["freeFlowSpeed"]
    current_time = flow["currentTravelTime"]
    free_flow_time = flow["freeFlowTravelTime"]

    speed_loss_pct = (
        (free_flow_speed - current_speed)
        / free_flow_speed
        * 100
    )

    delay_seconds = current_time - free_flow_time
    travel_time_ratio = current_time / free_flow_time

    return {
        "openlr": flow.get("openlr"),
        "current_speed": current_speed,
        "free_flow_speed": free_flow_speed,
        "current_travel_time": current_time,
        "free_flow_travel_time": free_flow_time,
        "delay_seconds": delay_seconds,
        "speed_loss_pct": speed_loss_pct,
        "travel_time_ratio": travel_time_ratio,
        "confidence": flow.get("confidence"),
        "road_closure": flow.get("roadClosure"),
    }


def crear_captura(api_key):
    """Genera una captura completa de los seis puntos."""
    collection_timestamp = datetime.now(timezone.utc).isoformat()

    observaciones = []

    for monitor_point_id, punto in PUNTOS_MONITOREO.items():
        resultado = consultar_segmento(
            punto["lat"],
            punto["lon"],
            api_key,
        )

        observaciones.append({
            "timestamp": collection_timestamp,
            "corridor_id": CORRIDOR_ID,
            "monitor_point_id": monitor_point_id,
            "segment_id": punto["segment_id"],
            "openlr": resultado["openlr"],
            "current_speed": resultado["current_speed"],
            "free_flow_speed": resultado["free_flow_speed"],
            "current_travel_time": resultado["current_travel_time"],
            "free_flow_travel_time": resultado["free_flow_travel_time"],
            "delay_seconds": resultado["delay_seconds"],
            "speed_loss_pct": resultado["speed_loss_pct"],
            "travel_time_ratio": resultado["travel_time_ratio"],
            "confidence": resultado["confidence"],
            "road_closure": resultado["road_closure"],
        })

    df = pd.DataFrame(observaciones)

    df["observation_id"] = (
        df["timestamp"].astype(str)
        + "_"
        + df["monitor_point_id"]
    )

    return df[
        [
            "observation_id",
            "timestamp",
            "corridor_id",
            "monitor_point_id",
            "segment_id",
            "openlr",
            "current_speed",
            "free_flow_speed",
            "current_travel_time",
            "free_flow_travel_time",
            "delay_seconds",
            "speed_loss_pct",
            "travel_time_ratio",
            "confidence",
            "road_closure",
        ]
    ]


def validar_captura(df):
    """Valida las reglas mínimas de calidad de una captura."""
    problemas = []

    if len(df) != len(PUNTOS_MONITOREO):
        problemas.append(
            f"Se esperaban {len(PUNTOS_MONITOREO)} registros; "
            f"se obtuvieron {len(df)}."
        )

    if df["monitor_point_id"].nunique() != len(PUNTOS_MONITOREO):
        problemas.append("No están presentes los 6 monitor points únicos.")

    if df["observation_id"].nunique() != len(df):
        problemas.append("Existen observation_id duplicados.")

    if df.isnull().sum().sum() > 0:
        problemas.append("Existen valores nulos.")

    if (df["current_speed"] > df["free_flow_speed"]).any():
        problemas.append("current_speed > free_flow_speed.")

    if (df["delay_seconds"] < 0).any():
        problemas.append("Existen delay_seconds negativos.")

    if (
        (df["speed_loss_pct"] < 0)
        | (df["speed_loss_pct"] > 100)
    ).any():
        problemas.append("speed_loss_pct fuera de rango 0-100.")

    if (df["travel_time_ratio"] < 1).any():
        problemas.append("travel_time_ratio < 1.")

    if (
        (df["confidence"] < 0)
        | (df["confidence"] > 1)
    ).any():
        problemas.append("confidence fuera de rango 0-1.")

    if (
        (df["current_travel_time"] <= 0)
        | (df["free_flow_travel_time"] <= 0)
    ).any():
        problemas.append("Tiempo de viaje inválido.")

    if not df["corridor_id"].eq(CORRIDOR_ID).all():
        problemas.append("Corridor ID inválido.")

    if not df["monitor_point_id"].isin(
        PUNTOS_MONITOREO.keys()
    ).all():
        problemas.append("Monitor Point ID inválido.")

    segmentos_validos = {
        punto["segment_id"]
        for punto in PUNTOS_MONITOREO.values()
    }

    if not df["segment_id"].isin(segmentos_validos).all():
        problemas.append("Segment ID invalido.")

    if problemas:
        raise ValueError(
            "DATA QUALITY: REVISAR-\n"
            + "\n-".join(problemas)
        )

    return True


def guardar_historico_traffic(df_nueva_captura, archivo):
    """Agrega una captura al histórico y elimina duplicados."""
    directorio = os.path.dirname(archivo)

    if directorio:
        os.makedirs(directorio, exist_ok=True)

    if os.path.exists(archivo):
        df_historico = pd.read_csv(archivo)
    else:
        df_historico = pd.DataFrame()

    df_historico = pd.concat(
        [df_historico, df_nueva_captura],
        ignore_index=True,
    )

    df_historico = df_historico.drop_duplicates(
        subset=["observation_id"]
    )

    df_historico = df_historico.sort_values(
        by=["timestamp", "monitor_point_id"]
    ).reset_index(drop=True)

    df_historico.to_csv(
        archivo,
        index=False,
        encoding="utf-8",
    )

    return df_historico


def main():
    api_key = os.getenv("TOMTOM_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No se encontró TOMTOM_API_KEY en las variables de entorno."
        )

    print("================================")
    print("PACHUCA MOBILITY — COLLECTOR")
    print("================================")

    print()
    print("Generando nueva captura...")

    df_captura = crear_captura(api_key)

    print("Registros:", len(df_captura))
    print(
        "Timestamps:",
        df_captura["timestamp"].nunique(),
    )
    print(
        "Observation IDs únicos:",
        df_captura["observation_id"].nunique(),
    )

    print()
    print("Validando Data Quality...")

    validar_captura(df_captura)

    print("DATA QUALITY: OK")

    print()
    print("Persistiendo histórico...")

    df_historico = guardar_historico_traffic(
        df_captura,
        ARCHIVO_TRAFFIC,
    )

    print()
    print("Histórico actualizado:")
    print("Registros:", len(df_historico))
    print(
        "Capturas:",
        df_historico["timestamp"].nunique(),
    )
    print(
        "IDs únicos:",
        df_historico["observation_id"].nunique(),
    )

    print()
    print("RESULTADO: OK")


if __name__ == "__main__":
    main()
