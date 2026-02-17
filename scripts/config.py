"""
config.py — Shared configuration for the analysis pipeline
============================================================
All paths are relative to this file's location, ensuring portability.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (portable: relative to this file)
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent          # projecting/
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"

# GLORIA MRIO database paths — adjust to your local installation.
# Download from https://ielab.info (Lenzen et al., 2017; 2022).
# Only required for script 01_build_tensor.py (tensor construction).
# Scripts 02-04 use intermediate outputs already included in data/.
GLORIA_META_DIR = ROOT_DIR.parent.parent / "metadata"   # default: ../../metadata/
GLORIA_TQ_DIR = ROOT_DIR.parent.parent / "TQ"           # default: ../../TQ/

# Backwards-compatible aliases used by 01_build_tensor.py
META_DIR = GLORIA_META_DIR
TQ_DIR = GLORIA_TQ_DIR

# Ensure output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Country groups
# ---------------------------------------------------------------------------
MERCOSUR = {"ARG": "Argentina", "BRA": "Brazil", "PRY": "Paraguay", "URY": "Uruguay"}
EU27 = {
    "AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria", "HRV": "Croatia",
    "CYP": "Cyprus", "CZE": "Czech Republic", "DNK": "Denmark", "EST": "Estonia",
    "FIN": "Finland", "FRA": "France", "DEU": "Germany", "GRC": "Greece",
    "HUN": "Hungary", "IRL": "Ireland", "ITA": "Italy", "LVA": "Latvia",
    "LTU": "Lithuania", "LUX": "Luxembourg", "MLT": "Malta", "NLD": "Netherlands",
    "POL": "Poland", "PRT": "Portugal", "ROU": "Romania", "SVK": "Slovakia",
    "SVN": "Slovenia", "ESP": "Spain", "SWE": "Sweden",
}
ALL_COUNTRIES = {**MERCOSUR, **EU27}
BLOC = {k: "MERCOSUR" for k in MERCOSUR} | {k: "EU27" for k in EU27}

# ---------------------------------------------------------------------------
# NTF component labels (shared across scripts 02, 03, 04)
# ---------------------------------------------------------------------------
COMP_LABELS = {
    0: "Agricultural crops",
    1: "Forestry & wood",
    2: "Fishery & aquatic",
    3: "Non-metallic minerals",
    4: "Livestock & grazing",
    5: "Metal ores",
}

# ---------------------------------------------------------------------------
# MFA colour palette
# ---------------------------------------------------------------------------
MFA_COLORS = {
    "Biomass": "#27ae60",
    "Metal ores": "#8e44ad",
    "Non-metallic minerals": "#f39c12",
    "Fossil fuels": "#2c3e50",
}
MFA_ORDER = ["Biomass", "Metal ores", "Non-metallic minerals", "Fossil fuels"]

BLOC_COLORS = {"MERCOSUR": "#e74c3c", "EU27": "#3498db"}

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
YEAR_RANGE = (1990, 2022)     # observed data window
OPTIMAL_K = 6                 # NTF rank (set after rank selection)
N_BOOTSTRAP = 1000            # bootstrap replications for projection
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Trade agreement shock parameters (from published tariff schedules, Dec 2024)
# Keys match NTF component indices (0-based)
# ---------------------------------------------------------------------------
AGREEMENT_SHOCKS = {
    0: {"mercosur_export_boost": 0.12, "eu_export_boost": 0.08,
        "description": "Agricultural crops: EU quotas for sugar/rice/cereals; MERCOSUR opens processed foods"},
    1: {"mercosur_export_boost": 0.06, "eu_export_boost": 0.04,
        "description": "Forestry: moderate liberalisation both ways"},
    2: {"mercosur_export_boost": 0.08, "eu_export_boost": 0.03,
        "description": "Fishery: EU quota expansion for MERCOSUR shrimp/fish"},
    3: {"mercosur_export_boost": 0.02, "eu_export_boost": 0.02,
        "description": "Non-metallic minerals: minimal trade impact (low tradability)"},
    4: {"mercosur_export_boost": 0.15, "eu_export_boost": 0.10,
        "description": "Livestock: EU beef quota (99,000t); MERCOSUR opens dairy/pork"},
    5: {"mercosur_export_boost": 0.10, "eu_export_boost": 0.05,
        "description": "Metal ores: critical minerals clause; EU machinery access"},
}

# Agreement timeline
AGREEMENT_YEAR = 2027         # assumed year agreement enters into force
PHASE_IN_YEARS = 5            # years for full implementation
PROJECTION_END = 2035         # exclusive upper bound (last projected year = 2034)
