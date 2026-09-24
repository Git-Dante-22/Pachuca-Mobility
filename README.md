
# 🚦 Pachuca Mobility

Plataforma de datos para el análisis de movilidad urbana en Pachuca, Hidalgo.

El proyecto busca integrar diferentes fuentes de datos relacionadas con movilidad, tráfico, infraestructura vial, accidentes y condiciones ambientales para construir una plataforma orientada al análisis histórico, generación de indicadores y futuros modelos de analítica y Machine Learning.

---

## 🎯 Objetivo

Construir una plataforma de datos que permita analizar la movilidad en Pachuca mediante la integración de diferentes fuentes de información.

La arquitectura propuesta busca evolucionar desde la captura de datos hacia procesos de:

- Ingesta de datos
- Almacenamiento histórico
- Limpieza y transformación
- Integración de información geoespacial
- Generación de KPIs
- Visualización
- Analítica avanzada
- Machine Learning

---

## 🏗️ Arquitectura

La arquitectura conceptual del proyecto es:

TomTom Traffic + OpenStreetMap + INEGI + Open-Meteo + otras fuentes
                         ↓
                      Python
                         ↓
                     Bronze
                         ↓
                     Silver
                         ↓
                      Gold
                    ↙       ↘
              Power BI       ML

Actualmente el proyecto se encuentra principalmente en las etapas de captura y almacenamiento de datos de tráfico.

---

## 🚗 Tráfico en tiempo real

La primera fuente implementada es TomTom Traffic API.

Actualmente se monitorea el corredor:

**Boulevard Felipe Ángeles – México-Pachuca**

Se utilizan 6 puntos de monitoreo:

| ID | Punto |
|---|---|
| MP001 | Unidad Deportiva |
| MP002 | SAT |
| MP003 | ADO Villas |
| MP004 | Puente a Av. las Torres |
| MP005 | Puente Matilde |
| MP006 | Salida Pachuca-CDMX |

Los puntos están asociados a 3 segmentos internos del corredor:

- S001
- S002
- S003

---

## 📊 Datos capturados

Cada observación de tráfico contiene información como:

- Timestamp
- Corridor ID
- Monitor Point ID
- Segment ID
- OpenLR
- Velocidad actual
- Velocidad de flujo libre
- Tiempo de viaje actual
- Tiempo de viaje en flujo libre
- Retraso en segundos
- Pérdida de velocidad
- Travel Time Ratio
- Confidence
- Road Closure

Estos datos permiten construir posteriormente indicadores de congestión y comportamiento histórico del corredor.

---

## 🧪 Data Quality

El collector ejecuta validaciones antes de persistir los datos.

Entre las validaciones implementadas se encuentran:

- Número esperado de registros
- Puntos de monitoreo válidos
- IDs únicos
- Valores no nulos
- Velocidad actual
- Velocidad de flujo libre
- Retraso
- Pérdida de velocidad
- Travel Time Ratio
- Confidence
- Tiempos de viaje positivos
- Identificadores de corredor
- Identificadores de puntos de monitoreo
- Identificadores de segmentos

Una captura únicamente se persiste cuando las validaciones son satisfactorias.

---

## 🔄 Collector

El archivo `collector.py` permite ejecutar una captura independiente del notebook de desarrollo.

Flujo:

```text
TomTom API
    ↓
collector.py
    ↓
6 puntos de monitoreo
    ↓
Data Quality
    ↓
CSV histórico
