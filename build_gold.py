import os
import glob
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

SILVER_PATH = "data/silver/traffic/*.csv"

GOLD_DIR = "data/gold/traffic"

GOLD_MONITOR_FILE = os.path.join(
    GOLD_DIR,
    "gold_traffic_30min_monitor.csv"
)

GOLD_SEGMENT_FILE = os.path.join(
    GOLD_DIR,
    "gold_traffic_30min_segment.csv"
)

GOLD_EPISODE_FILE = os.path.join(
    GOLD_DIR,
    "gold_congestion_episode.csv"
)

TIMEZONE_LOCAL = "America/Mexico_City"

# Regla inicial del proyecto:
# un intervalo se considera congestionado
# cuando la pérdida de velocidad es >= 30%
CONGESTION_THRESHOLD = 30

# Un episodio requiere al menos
# 2 intervalos consecutivos de 30 minutos
MIN_EPISODE_INTERVALS = 2


# ============================================================
# CARGAR SILVER
# ============================================================

def cargar_silver():

    archivos = glob.glob(SILVER_PATH)

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos Silver en: {SILVER_PATH}"
        )

    print("Archivos Silver encontrados:")

    for archivo in sorted(archivos):
        print(f"  - {archivo}")

    dataframes = []

    for archivo in sorted(archivos):

        df = pd.read_csv(archivo)

        dataframes.append(df)

    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(
        f"\nFilas Silver antes de deduplicar: {len(df)}"
    )

    if "observation_id" in df.columns:

        df = df.drop_duplicates(
            subset=["observation_id"]
        ).reset_index(drop=True)

    print(
        f"Filas Silver después de deduplicar: {len(df)}"
    )

    return df


# ============================================================
# PREPARAR DATOS
# ============================================================

def preparar_datos(df):

    columnas_requeridas = [
        "observation_id",
        "timestamp",
        "corridor_id",
        "monitor_point_id",
        "segment_id",
        "current_speed",
        "free_flow_speed",
        "current_travel_time",
        "free_flow_travel_time",
        "delay_seconds",
        "speed_loss_pct",
        "travel_time_ratio",
        "confidence",
        "road_closure"
    ]

    columnas_faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in df.columns
    ]

    if columnas_faltantes:

        raise ValueError(
            "Faltan columnas requeridas en Silver: "
            + ", ".join(columnas_faltantes)
        )

    # Timestamp almacenado en UTC
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # Convertir a hora local de México
    df["timestamp_local"] = (
        df["timestamp"]
        .dt.tz_convert(TIMEZONE_LOCAL)
    )

    # Agrupar en intervalos de 30 minutos
    df["timestamp_30min"] = (
        df["timestamp_local"]
        .dt.floor("30min")
    )

    # Dimensiones temporales

    df["date_local"] = (
        df["timestamp_30min"]
        .dt.date
    )

    df["hour_local"] = (
        df["timestamp_30min"]
        .dt.hour
    )

    df["day_of_week"] = (
        df["timestamp_30min"]
        .dt.day_name()
    )

    df["day_type"] = (
        df["timestamp_30min"]
        .dt.dayofweek
        .map(
            lambda x:
            "Weekend"
            if x >= 5
            else "Weekday"
        )
    )

    return df


# ============================================================
# GOLD — TRAFFIC 30 MIN POR MONITOR
# ============================================================

def construir_gold_monitor(df):

    columnas_grupo = [
        "timestamp_30min",
        "corridor_id",
        "monitor_point_id",
        "segment_id"
    ]

    gold_monitor = (
        df
        .groupby(columnas_grupo)
        .agg(

            avg_current_speed=(
                "current_speed",
                "mean"
            ),

            avg_free_flow_speed=(
                "free_flow_speed",
                "mean"
            ),

            avg_current_travel_time=(
                "current_travel_time",
                "mean"
            ),

            avg_free_flow_travel_time=(
                "free_flow_travel_time",
                "mean"
            ),

            avg_delay_seconds=(
                "delay_seconds",
                "mean"
            ),

            avg_speed_loss_pct=(
                "speed_loss_pct",
                "mean"
            ),

            avg_travel_time_ratio=(
                "travel_time_ratio",
                "mean"
            ),

            avg_confidence=(
                "confidence",
                "mean"
            ),

            min_current_speed=(
                "current_speed",
                "min"
            ),

            max_current_speed=(
                "current_speed",
                "max"
            ),

            std_current_speed=(
                "current_speed",
                "std"
            ),

            observations=(
                "observation_id",
                "count"
            ),

            road_closure=(
                "road_closure",
                "max"
            )
        )
        .reset_index()
    )

    gold_monitor["std_current_speed"] = (
        gold_monitor["std_current_speed"]
        .fillna(0)
    )

    gold_monitor["date_local"] = (
        gold_monitor["timestamp_30min"]
        .dt.date
    )

    gold_monitor["hour_local"] = (
        gold_monitor["timestamp_30min"]
        .dt.hour
    )

    gold_monitor["day_of_week"] = (
        gold_monitor["timestamp_30min"]
        .dt.day_name()
    )

    gold_monitor["day_type"] = (
        gold_monitor["timestamp_30min"]
        .dt.dayofweek
        .map(
            lambda x:
            "Weekend"
            if x >= 5
            else "Weekday"
        )
    )

    gold_monitor = gold_monitor.sort_values(
        by=[
            "timestamp_30min",
            "monitor_point_id"
        ]
    ).reset_index(drop=True)

    return gold_monitor


# ============================================================
# GOLD — TRAFFIC 30 MIN POR SEGMENTO
# ============================================================

def construir_gold_segment(df):

    columnas_grupo = [
        "timestamp_30min",
        "corridor_id",
        "segment_id"
    ]

    gold_segment = (
        df
        .groupby(columnas_grupo)
        .agg(

            avg_current_speed=(
                "current_speed",
                "mean"
            ),

            avg_free_flow_speed=(
                "free_flow_speed",
                "mean"
            ),

            avg_current_travel_time=(
                "current_travel_time",
                "mean"
            ),

            avg_free_flow_travel_time=(
                "free_flow_travel_time",
                "mean"
            ),

            avg_delay_seconds=(
                "delay_seconds",
                "mean"
            ),

            avg_speed_loss_pct=(
                "speed_loss_pct",
                "mean"
            ),

            avg_travel_time_ratio=(
                "travel_time_ratio",
                "mean"
            ),

            avg_confidence=(
                "confidence",
                "mean"
            ),

            min_current_speed=(
                "current_speed",
                "min"
            ),

            max_current_speed=(
                "current_speed",
                "max"
            ),

            std_current_speed=(
                "current_speed",
                "std"
            ),

            observations=(
                "observation_id",
                "count"
            ),

            road_closure=(
                "road_closure",
                "max"
            ),

            monitors=(
                "monitor_point_id",
                "nunique"
            )
        )
        .reset_index()
    )

    gold_segment["std_current_speed"] = (
        gold_segment["std_current_speed"]
        .fillna(0)
    )

    gold_segment["date_local"] = (
        gold_segment["timestamp_30min"]
        .dt.date
    )

    gold_segment["hour_local"] = (
        gold_segment["timestamp_30min"]
        .dt.hour
    )

    gold_segment["day_of_week"] = (
        gold_segment["timestamp_30min"]
        .dt.day_name()
    )

    gold_segment["day_type"] = (
        gold_segment["timestamp_30min"]
        .dt.dayofweek
        .map(
            lambda x:
            "Weekend"
            if x >= 5
            else "Weekday"
        )
    )

    gold_segment = gold_segment.sort_values(
        by=[
            "segment_id",
            "timestamp_30min"
        ]
    ).reset_index(drop=True)

    return gold_segment


# ============================================================
# GOLD — EPISODIOS DE CONGESTIÓN
# ============================================================

def construir_gold_episodes(gold_segment):

    df = gold_segment.copy()

    # --------------------------------------------------------
    # 1. Identificar intervalos congestionados
    # --------------------------------------------------------

    df["is_congested"] = (
        df["avg_speed_loss_pct"]
        >= CONGESTION_THRESHOLD
    )

    # --------------------------------------------------------
    # 2. Timestamp anterior dentro de cada segmento
    # --------------------------------------------------------

    df["prev_timestamp"] = (
        df
        .groupby("segment_id")["timestamp_30min"]
        .shift(1)
    )

    # --------------------------------------------------------
    # 3. Determinar si el intervalo es consecutivo
    # --------------------------------------------------------

    df["is_consecutive"] = (
        (
            df["timestamp_30min"]
            - df["prev_timestamp"]
        )
        == pd.Timedelta(minutes=30)
    )

    # --------------------------------------------------------
    # 4. Saber si el intervalo anterior estaba congestionado
    # --------------------------------------------------------

    previous_congested = (
        df
        .groupby("segment_id")["is_congested"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # 5. Detectar inicio de nuevo episodio
    # --------------------------------------------------------

    df["new_episode"] = (
        df["is_congested"]
        & (
            ~df["is_consecutive"]
            | ~previous_congested
        )
    )

    # --------------------------------------------------------
    # 6. Crear identificador interno del grupo
    # --------------------------------------------------------

    df["episode_group"] = (
        df
        .groupby("segment_id")["new_episode"]
        .cumsum()
    )

    # --------------------------------------------------------
    # 7. Resumir grupos congestionados
    # --------------------------------------------------------

    resumen_grupos = (
        df[df["is_congested"]]
        .groupby(
            [
                "segment_id",
                "episode_group"
            ]
        )
        .agg(

            start_time=(
                "timestamp_30min",
                "min"
            ),

            end_time=(
                "timestamp_30min",
                "max"
            ),

            intervals=(
                "timestamp_30min",
                "count"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 8. Conservar solamente grupos >= 2 intervalos
    # --------------------------------------------------------

    claves_validas = (
        resumen_grupos[
            resumen_grupos["intervals"]
            >= MIN_EPISODE_INTERVALS
        ][
            [
                "segment_id",
                "episode_group"
            ]
        ]
        .copy()
    )

    # --------------------------------------------------------
    # 9. Filtrar registros pertenecientes a episodios válidos
    #
    # Importante:
    # el episode_group se reinicia por segmento.
    # Por eso el merge utiliza ambas columnas.
    # --------------------------------------------------------

    episodios_base = (
        df[df["is_congested"]]
        .merge(
            claves_validas,
            on=[
                "segment_id",
                "episode_group"
            ],
            how="inner"
        )
    )

    # --------------------------------------------------------
    # 10. Calcular métricas del episodio
    # --------------------------------------------------------

    episodios = (
        episodios_base
        .groupby(
            [
                "segment_id",
                "episode_group"
            ]
        )
        .agg(

            start_time=(
                "timestamp_30min",
                "min"
            ),

            end_time=(
                "timestamp_30min",
                "max"
            ),

            observations=(
                "timestamp_30min",
                "count"
            ),

            avg_speed_loss_pct=(
                "avg_speed_loss_pct",
                "mean"
            ),

            max_speed_loss_pct=(
                "avg_speed_loss_pct",
                "max"
            ),

            avg_delay_seconds=(
                "avg_delay_seconds",
                "mean"
            ),

            max_delay_seconds=(
                "avg_delay_seconds",
                "max"
            ),

            avg_travel_time_ratio=(
                "avg_travel_time_ratio",
                "mean"
            ),

            max_travel_time_ratio=(
                "avg_travel_time_ratio",
                "max"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 11. Duración
    # --------------------------------------------------------

    episodios["duration_minutes"] = (
        episodios["observations"]
        * 30
    )

    # --------------------------------------------------------
    # 12. Crear episode_id
    # --------------------------------------------------------

    episodios["episode_id"] = (
        episodios["segment_id"]
        + "_E"
        + (
            episodios
            .groupby("segment_id")
            .cumcount()
            .add(1)
            .astype(str)
            .str.zfill(3)
        )
    )

    # --------------------------------------------------------
    # 13. Orden final de columnas
    # --------------------------------------------------------

    episodios = episodios[
        [
            "episode_id",
            "segment_id",
            "start_time",
            "end_time",
            "duration_minutes",
            "observations",
            "avg_speed_loss_pct",
            "max_speed_loss_pct",
            "avg_delay_seconds",
            "max_delay_seconds",
            "avg_travel_time_ratio",
            "max_travel_time_ratio"
        ]
    ]

    episodios = (
        episodios
        .sort_values(
            by=[
                "segment_id",
                "start_time"
            ]
        )
        .reset_index(drop=True)
    )

    return episodios


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PACHUCA MOBILITY - BUILD GOLD")
    print("=" * 60)

    # --------------------------------------------------------
    # Crear directorio Gold
    # --------------------------------------------------------

    os.makedirs(
        GOLD_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Cargar Silver
    # --------------------------------------------------------

    df_silver = cargar_silver()

    # --------------------------------------------------------
    # 2. Preparar datos
    # --------------------------------------------------------

    df = preparar_datos(
        df_silver
    )

    print(
        f"\nFilas preparadas: {len(df)}"
    )

    print("\nRango temporal local:")

    print(
        f"  Inicio: {df['timestamp_30min'].min()}"
    )

    print(
        f"  Fin:    {df['timestamp_30min'].max()}"
    )

    # --------------------------------------------------------
    # 3. Construir Gold monitor
    # --------------------------------------------------------

    print("\nConstruyendo Gold monitor...")

    gold_monitor = construir_gold_monitor(
        df
    )

    gold_monitor.to_csv(
        GOLD_MONITOR_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"  Filas: {len(gold_monitor)}"
    )

    print(
        f"  Archivo: {GOLD_MONITOR_FILE}"
    )

    # --------------------------------------------------------
    # 4. Construir Gold segment
    # --------------------------------------------------------

    print("\nConstruyendo Gold segment...")

    gold_segment = construir_gold_segment(
        df
    )

    gold_segment.to_csv(
        GOLD_SEGMENT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"  Filas: {len(gold_segment)}"
    )

    print(
        f"  Archivo: {GOLD_SEGMENT_FILE}"
    )

    # --------------------------------------------------------
    # 5. Construir episodios de congestión
    # --------------------------------------------------------

    print(
        "\nConstruyendo Gold congestion episodes..."
    )

    episodios = construir_gold_episodes(
        gold_segment
    )

    episodios.to_csv(
        GOLD_EPISODE_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"  Episodios: {len(episodios)}"
    )

    print(
        f"  Archivo: {GOLD_EPISODE_FILE}"
    )

    # --------------------------------------------------------
    # 6. Validaciones
    # --------------------------------------------------------

    print("\nValidaciones:")

    print(
        "  Gold monitor sin nulos:",
        gold_monitor.isnull().sum().sum() == 0
    )

    print(
        "  Gold segment sin nulos:",
        gold_segment.isnull().sum().sum() == 0
    )

    print(
        "  Episodios >= 2 intervalos:",
        (
            episodios["observations"]
            >= MIN_EPISODE_INTERVALS
        ).all()
    )

    print(
        "  Episodios con speed loss >= 30%:",
        (
            episodios["avg_speed_loss_pct"]
            >= CONGESTION_THRESHOLD
        ).all()
    )

    print(
        "  Duraciones correctas:",
        (
            episodios["duration_minutes"]
            == episodios["observations"] * 30
        ).all()
    )

    print(
        "  Episodios sin nulos:",
        episodios.isnull().sum().sum() == 0
    )

    print("\n" + "=" * 60)
    print("GOLD GENERADO CORRECTAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()
