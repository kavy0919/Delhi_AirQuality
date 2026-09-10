"""
advisory_rules.py
==================
Deterministic rule engine mapping predicted PM2.5 / AQI to Delhi-NCR's
official Graded Response Action Plan (GRAP) stages, plus municipal policy
actions and persona-specific citizen health advisories.

Thresholds follow CAQM's (Commission for Air Quality Management) published
GRAP revision (2023) PM2.5 / AQI bands:

    Stage I    'Poor'      AQI 201-300   | PM2.5  61-120 ug/m3
    Stage II   'Very Poor' AQI 301-400   | PM2.5 121-250 ug/m3
    Stage III  'Severe'    AQI 401-450   | PM2.5 251-350 ug/m3
    Stage IV   'Severe+'   AQI  >450     | PM2.5  >350   ug/m3

Below Stage I (AQI <= 200 / PM2.5 <= 60) is treated as a baseline
"Satisfactory/Moderate — No GRAP Action" band so the classifier always
returns a well-defined result.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


# --------------------------------------------------------------------------
# STAGE DEFINITION
# --------------------------------------------------------------------------
@dataclass
class GrapStage:
    stage_id: int                 # 0 = below GRAP, 1-4 = Stage I-IV
    stage_label: str
    aqi_category: str
    color: str                    # for UI badges / map markers
    pm25_range: str
    aqi_range: str
    policy_actions: List[str] = field(default_factory=list)
    persona_advisories: Dict[str, List[str]] = field(default_factory=dict)


_STAGES: List[GrapStage] = [
    GrapStage(
        stage_id=0,
        stage_label="Below GRAP",
        aqi_category="Satisfactory / Moderate",
        color="#2ECC71",  # green
        pm25_range="0 - 60 µg/m³",
        aqi_range="0 - 200",
        policy_actions=[
            "No GRAP restrictions currently in force.",
            "Routine monitoring of pollution sources continues.",
            "Continue mechanized road sweeping and water sprinkling on a normal schedule.",
        ],
        persona_advisories={
            "Vulnerable / Asthmatics": [
                "Air quality is generally safe; continue regular medication as prescribed.",
                "Keep rescue inhalers handy if you have a known respiratory condition.",
            ],
            "Morning Walkers / Athletes": [
                "Good day for outdoor exercise and jogging.",
                "Early morning or evening slots are fine.",
            ],
            "General Public": [
                "No special precautions needed.",
                "Enjoy normal outdoor activities.",
            ],
        },
    ),
    GrapStage(
        stage_id=1,
        stage_label="Stage I — Poor",
        aqi_category="Poor",
        color="#F1C40F",  # yellow
        pm25_range="61 - 120 µg/m³",
        aqi_range="201 - 300",
        policy_actions=[
            "Strict enforcement of dust control at all construction & demolition (C&D) sites.",
            "Ban on garbage burning in open areas; heavy fines for violations.",
            "Mechanized sweeping and water sprinkling on roads intensified, especially at hotspots.",
            "Regular inspection of pollution-control devices at brick kilns and industries.",
            "Traffic police to strictly enforce PUC (Pollution Under Control) norms at intersections.",
        ],
        persona_advisories={
            "Vulnerable / Asthmatics": [
                "Reduce prolonged or heavy outdoor exertion.",
                "Keep prescribed inhalers/medication accessible at all times.",
                "Consider wearing an N95 mask when stepping out for extended periods.",
            ],
            "Morning Walkers / Athletes": [
                "Shift intense workouts indoors or to well-ventilated gyms where possible.",
                "If walking outside, prefer greener areas away from traffic corridors.",
            ],
            "General Public": [
                "Sensitive individuals (children, elderly) should limit prolonged outdoor exertion.",
                "Keep windows closed during peak traffic hours (morning/evening).",
            ],
        },
    ),
    GrapStage(
        stage_id=2,
        stage_label="Stage II — Very Poor",
        aqi_category="Very Poor",
        color="#E67E22",  # orange
        pm25_range="121 - 250 µg/m³",
        aqi_range="301 - 400",
        policy_actions=[
            "Diesel generator sets (except emergency/essential services) banned.",
            "Parking fees increased to discourage private vehicle usage.",
            "Frequency of CNG/electric bus and metro services increased.",
            "Stone crushers and mining-related activities in NCR suspended.",
            "Hotspot-specific enforcement teams deployed with additional anti-smog guns.",
        ],
        persona_advisories={
            "Vulnerable / Asthmatics": [
                "Avoid outdoor activity altogether; stay indoors with air purifiers if available.",
                "Keep emergency medication readily accessible and monitor symptoms closely.",
                "Consult a doctor promptly if experiencing breathlessness or persistent coughing.",
            ],
            "Morning Walkers / Athletes": [
                "Postpone outdoor exercise entirely; switch to indoor workouts.",
                "Avoid early-morning outdoor activity when pollutant concentration typically peaks.",
            ],
            "General Public": [
                "Wear an N95/N99 mask if you must step outside.",
                "Children and elderly should avoid outdoor exposure as much as possible.",
                "Keep indoor air purifiers running; seal gaps in doors/windows.",
            ],
        },
    ),
    GrapStage(
        stage_id=3,
        stage_label="Stage III — Severe",
        aqi_category="Severe",
        color="#E74C3C",  # red
        pm25_range="251 - 350 µg/m³",
        aqi_range="401 - 450",
        policy_actions=[
            "Ban on all non-essential construction and demolition activity across NCR.",
            "BS-III petrol and BS-IV diesel four-wheelers banned from plying in Delhi.",
            "Physical classes for children up to Class V suspended; shift to online mode.",
            "Odd-even vehicle rationing scheme considered by state governments.",
            "Additional water sprinkling, smog guns, and anti-smog towers activated across hotspots.",
        ],
        persona_advisories={
            "Vulnerable / Asthmatics": [
                "Stay indoors at all times; avoid any outdoor exposure.",
                "Use a certified air purifier and keep windows/doors sealed.",
                "Seek immediate medical attention for any respiratory distress.",
            ],
            "Morning Walkers / Athletes": [
                "All outdoor exercise should be stopped completely.",
                "Even brief outdoor exposure should be minimized.",
            ],
            "General Public": [
                "Avoid stepping outside unless absolutely necessary; wear an N95/N99 mask if you do.",
                "Schools advised to suspend outdoor sports and assemblies.",
                "Elderly, children, and those with pre-existing conditions are at high risk — stay indoors.",
            ],
        },
    ),
    GrapStage(
        stage_id=4,
        stage_label="Stage IV — Severe+",
        aqi_category="Severe Plus (Emergency)",
        color="#7B241C",  # dark maroon
        pm25_range="> 350 µg/m³",
        aqi_range="> 450",
        policy_actions=[
            "Entry of trucks (non-essential goods) into Delhi banned, except CNG/electric/BS-VI.",
            "All non-essential construction and demolition activity completely banned (state-wide).",
            "State governments may decide on closure of physical classes for all schools.",
            "Odd-even vehicle rationing and work-from-home for government/private offices considered.",
            "District Magistrates empowered to take additional emergency local measures.",
        ],
        persona_advisories={
            "Vulnerable / Asthmatics": [
                "Do not step outside under any circumstances.",
                "Keep emergency contacts and medication ready; monitor oxygen levels if possible.",
                "Seek immediate hospital care for any breathing difficulty.",
            ],
            "Morning Walkers / Athletes": [
                "All outdoor physical activity must be avoided entirely.",
                "Indoor activity should also be minimized if indoor air quality is compromised.",
            ],
            "General Public": [
                "This is a public health emergency — remain indoors with air purifiers running.",
                "Avoid all non-essential travel outdoors.",
                "Follow official government advisories and local authority instructions closely.",
            ],
        },
    ),
]

_STAGE_BY_ID = {s.stage_id: s for s in _STAGES}


# --------------------------------------------------------------------------
# CLASSIFIER
# --------------------------------------------------------------------------
def _stage_from_pm25(pm25: float) -> int:
    if pm25 > 350:
        return 4
    if pm25 > 250:
        return 3
    if pm25 > 120:
        return 2
    if pm25 > 60:
        return 1
    return 0


def _stage_from_aqi(aqi: float) -> int:
    if aqi > 450:
        return 4
    if aqi > 400:
        return 3
    if aqi > 300:
        return 2
    if aqi > 200:
        return 1
    return 0


def classify_grap(pm25: Optional[float] = None, aqi: Optional[float] = None) -> GrapStage:
    """
    Classifies a GRAP stage from PM2.5 (µg/m3) and/or AQI. If both are
    provided, the MORE SEVERE (higher) of the two resulting stages is
    returned, matching CAQM's precautionary approach.

    Raises ValueError if neither value is supplied.
    """
    if pm25 is None and aqi is None:
        raise ValueError("classify_grap requires at least one of pm25 or aqi.")

    candidate_ids = []
    if pm25 is not None:
        candidate_ids.append(_stage_from_pm25(pm25))
    if aqi is not None:
        candidate_ids.append(_stage_from_aqi(aqi))

    stage_id = max(candidate_ids)
    return _STAGE_BY_ID[stage_id]


def get_all_stages() -> List[GrapStage]:
    """Returns all defined stages (0-4), useful for building legends."""
    return _STAGES


def stage_to_dict(stage: GrapStage) -> dict:
    """Convenience serializer for Streamlit / JSON use."""
    return {
        "stage_id": stage.stage_id,
        "stage_label": stage.stage_label,
        "aqi_category": stage.aqi_category,
        "color": stage.color,
        "pm25_range": stage.pm25_range,
        "aqi_range": stage.aqi_range,
        "policy_actions": stage.policy_actions,
        "persona_advisories": stage.persona_advisories,
    }


if __name__ == "__main__":
    # quick self-test
    for test_pm25 in [40, 90, 180, 300, 400]:
        s = classify_grap(pm25=test_pm25)
        print(f"PM2.5={test_pm25:>4} -> {s.stage_label:<22} ({s.aqi_category})")
