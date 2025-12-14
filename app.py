import streamlit as st
import pandas as pd
import requests
from functools import lru_cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AbrivioMVP/1.0; +https://abrivio.example)",
    "Accept": "application/json,text/plain,*/*",
}

# FARS/State numeric codes used in FARS extracts (includes PR=72).
# This guarantees the dropdown is limited to valid regions in the FARS schema even if definitions endpoint is blocked.
FARS_STATE_CODES = {
    "Alabama": 1, "Alaska": 2, "Arizona": 4, "Arkansas": 5, "California": 6, "Colorado": 8,
    "Connecticut": 9, "Delaware": 10, "District of Columbia": 11, "Florida": 12, "Georgia": 13,
    "Hawaii": 15, "Idaho": 16, "Illinois": 17, "Indiana": 18, "Iowa": 19, "Kansas": 20,
    "Kentucky": 21, "Louisiana": 22, "Maine": 23, "Maryland": 24, "Massachusetts": 25,
    "Michigan": 26, "Minnesota": 27, "Mississippi": 28, "Missouri": 29, "Montana": 30,
    "Nebraska": 31, "Nevada": 32, "New Hampshire": 33, "New Jersey": 34, "New Mexico": 35,
    "New York": 36, "North Carolina": 37, "North Dakota": 38, "Ohio": 39, "Oklahoma": 40,
    "Oregon": 41, "Pennsylvania": 42, "Rhode Island": 44, "South Carolina": 45,
    "South Dakota": 46, "Tennessee": 47, "Texas": 48, "Utah": 49, "Vermont": 50,
    "Virginia": 51, "Washington": 53, "West Virginia": 54, "Wisconsin": 55, "Wyoming": 56,
    "Puerto Rico": 72,
}

# ----------------------------
# Config
# ----------------------------
NHTSA_BASE = "https://crashviewer.nhtsa.dot.gov/CrashAPI"
CENSUS_BASE = "https://api.census.gov/data"

st.set_page_config(
    page_title="Abrivio | NHTSA Regional Safety Insights",
    layout="wide",
)

# ----------------------------
# Utilities
# ----------------------------
def http_get_json(url: str, params: dict):
    r = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=60
    )
    r.raise_for_status()
    return r.json()

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

# ----------------------------
# NHTSA: Definitions + Data (FARS 2010+ via CrashAPI)
# ----------------------------
@lru_cache(maxsize=64)
def nhtsa_variable_attributes(variable: str, case_year: int) -> pd.DataFrame:
    # CrashAPI definitions endpoint
    url = f"{NHTSA_BASE}/definitions/GetVariableAttributes"
    params = {"variable": variable, "caseYear": case_year, "format": "json"}
    j = http_get_json(url, params)
    return pd.DataFrame(j.get("Results", []))

@lru_cache(maxsize=64)
def nhtsa_fars_dataset(dataset: str, from_year: int, to_year: int, state: int) -> pd.DataFrame:
    # CrashAPI FARS dataset endpoint
    url = f"{NHTSA_BASE}/FARSData/GetFARSData"
    params = {
        "dataset": dataset,
        "FromYear": from_year,
        "ToYear": to_year,
        "State": state,
        "format": "json",
    }
    j = http_get_json(url, params)
    return pd.DataFrame(j.get("Results", []))

# ----------------------------
# Census: Population Estimates (state, year)
# ----------------------------
@lru_cache(maxsize=256)
def census_state_population(state_fips: int, year: int, census_api_key: str | None) -> int:
    """
    Census PEP endpoint format:
      /data/{year}/pep/population
    Variable naming is POP_{year} (see variables list in the endpoint docs). :contentReference[oaicite:1]{index=1}
    """
    url = f"{CENSUS_BASE}/{year}/pep/population"
    params = {
        "get": f"NAME,POP_{year}",
        "for": f"state:{state_fips:02d}",
    }
    if census_api_key:
        params["key"] = census_api_key

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    header, row = data[0], data[1]
    pop_col = header.index(f"POP_{year}")
    return int(row[pop_col])

# ----------------------------
# Analytics
# ----------------------------
def compute_fatalities(person_df: pd.DataFrame) -> pd.DataFrame:
    if person_df.empty:
        return person_df

    df = person_df.rename(columns={c: c.upper() for c in person_df.columns}).copy()
    df["INJ_SEV"] = to_num(df.get("INJ_SEV"))

    # Fatal injury (K) commonly corresponds to INJ_SEV == 4 in CrashAPI PERSON extracts.
    fatal = df[df["INJ_SEV"] == 4].copy()

    if "YEAR" in fatal.columns:
        fatal["YEAR"] = to_num(fatal["YEAR"])
    if "AGE" in fatal.columns:
        fatal["AGE"] = to_num(fatal["AGE"])

    return fatal

def pick_bac_series(fatal_df: pd.DataFrame) -> pd.Series | None:
    # Field names may vary by year; try common candidates.
    for c in ["BAC", "ALC_RES", "ALCRES", "ALCDET"]:
        if c in fatal_df.columns:
            return to_num(fatal_df[c])
    return None

def drug_involvement_percent(fatal_df: pd.DataFrame, drugs_df: pd.DataFrame | None) -> float | None:
    if drugs_df is None or drugs_df.empty or fatal_df.empty:
        return None

    f = fatal_df.copy()
    d = drugs_df.rename(columns={c: c.upper() for c in drugs_df.columns}).copy()

    keys = [k for k in ["ST_CASE", "VEH_NO", "PER_NO"] if k in f.columns and k in d.columns]
    if len(keys) < 2:
        return None

    m = f.merge(d, on=keys, how="left", suffixes=("", "_DRUG"))
    drug_cols = [c for c in m.columns if "DRUG" in c]
    if not drug_cols:
        return None

    has_drug = m[drug_cols].notna().any(axis=1)
    return float(has_drug.mean() * 100.0)

def fatality_rate_per_100k_by_year(fatal_df: pd.DataFrame, state_fips: int, from_year: int, to_year: int, census_api_key: str | None):
    rows = []
    for y in range(from_year, to_year + 1):
        deaths = int((fatal_df.get("YEAR", pd.Series([], dtype=float)) == y).sum()) if "YEAR" in fatal_df.columns else int(len(fatal_df))
        pop = census_state_population(state_fips, y, census_api_key)
        rate = (deaths / pop) * 100000.0
        rows.append({"year": y, "fatalities": deaths, "population": pop, "rate_per_100k": rate})
    table = pd.DataFrame(rows)
    return table, float(table["rate_per_100k"].mean())

# ----------------------------
# Demo Mode (offline preview)
# ----------------------------
def demo_data():
    fatal = pd.DataFrame({
        "YEAR": [2019, 2019, 2020, 2021, 2021, 2022, 2023],
        "AGE":  [22,   49,   36,   18,   67,   41,   29],
        "INJ_SEV": [4, 4, 4, 4, 4, 4, 4],
        "BAC":  [0.00, 0.12, 0.09, 0.00, 0.00, 0.08, 0.02],
        "ST_CASE": [1,2,3,4,5,6,7],
        "VEH_NO":  [1,1,1,1,1,1,1],
        "PER_NO":  [1,1,1,1,1,1,1],
    })
    drugs = pd.DataFrame({
        "ST_CASE": [2, 3, 6],
        "VEH_NO":  [1, 1, 1],
        "PER_NO":  [1, 1, 1],
        "DRUGRES": ["X", "X", "X"],
    })
    pop = {2019: 1000000, 2020: 1000000, 2021: 1000000, 2022: 1000000, 2023: 1000000}
    return fatal, drugs, pop

# ----------------------------
# UI
# ----------------------------
st.title("Abrivio | NHTSA Regional Safety Insights")
st.caption("MVP: NHTSA FARS (CrashAPI) + Census Population Estimates to answer 3 regional safety questions.")

with st.sidebar:
    st.header("Controls")
    demo_mode = st.toggle("Demo mode (no internet)", value=False)

    if not demo_mode:
        defs_year = st.number_input("NHTSA definitions year", min_value=2010, max_value=2024, value=2023, step=1)

        # Pull valid states from NHTSA definitions so dropdown never contains invalid regions
        try:
            state_defs = nhtsa_variable_attributes("state", int(defs_year))
            val_col = next((c for c in state_defs.columns if c.lower() in ["attributevalue", "value", "id"]), None)
            name_col = next((c for c in state_defs.columns if c.lower() in ["attributename", "name", "title", "description"]), None)

            if not val_col or not name_col:
                raise ValueError("Could not detect state definition columns.")

            state_defs = state_defs[[val_col, name_col]].dropna().drop_duplicates()
            state_defs[val_col] = to_num(state_defs[val_col]).dropna().astype(int)
            state_defs = state_defs.sort_values(name_col)
        
            label_to_code = dict(zip(state_defs[name_col], state_defs[val_col]))
            state_label = st.selectbox("Region (State)", list(label_to_code.keys()))
            state_code = int(label_to_code[state_label])
        
        except Exception:
            st.warning("NHTSA definitions endpoint blocked (403). Using built-in FARS state list fallback.")
            state_label = st.selectbox("Region (State)", sorted(FARS_STATE_CODES.keys()))
            state_code = int(FARS_STATE_CODES[state_label])


        from_year = st.number_input("From year", min_value=2010, max_value=2024, value=2019, step=1)
        to_year = st.number_input("To year", min_value=2010, max_value=2024, value=2023, step=1)

        st.subheader("Alcohol / Drugs logic")
        bac_threshold = st.selectbox(
            "Alcohol threshold",
            options=[0.0, 0.08],
            index=1,
            format_func=lambda x: "BAC > 0" if x == 0.0 else "BAC ≥ 0.08",
        )

        st.subheader("Census API key (optional)")
        st.caption("If you set a key in Streamlit secrets as CENSUS_API_KEY, it will be used automatically.")
        run = st.button("Run analysis", type="primary")
    else:
        state_label = st.selectbox("Region (Demo)", ["Demo State"])
        state_code = 1
        from_year = st.number_input("From year (Demo)", min_value=2019, max_value=2023, value=2019, step=1)
        to_year = st.number_input("To year (Demo)", min_value=2019, max_value=2023, value=2023, step=1)
        bac_threshold = st.selectbox(
            "Alcohol threshold (Demo)",
            options=[0.0, 0.08],
            index=1,
            format_func=lambda x: "BAC > 0" if x == 0.0 else "BAC ≥ 0.08",
        )
        run = st.button("Run analysis (Demo)", type="primary")

# Pull key from Streamlit secrets if present
census_key = None
try:
    census_key = st.secrets.get("CENSUS_API_KEY", None)
except Exception:
    census_key = None

if run:
    if from_year > to_year:
        st.error("From year must be <= To year.")
        st.stop()

    if demo_mode:
        fatal, drugs, pop = demo_data()

        # Override the population function in demo mode
        def demo_pop(state_fips, year, key=None):  # noqa: ARG001
            return pop[int(year)]
        globals()["census_state_population"] = lru_cache(maxsize=256)(demo_pop)

        person = fatal
        drugs_df = drugs
    else:
        with st.spinner("Loading NHTSA FARS PERSON (+ DRUGS best-effort) ..."):
            person = nhtsa_fars_dataset("PERSON", int(from_year), int(to_year), int(state_code))
            try:
                drugs_df = nhtsa_fars_dataset("DRUGS", int(from_year), int(to_year), int(state_code))
            except Exception:
                drugs_df = None

    fatal_df = compute_fatalities(person)

    with st.spinner("Computing fatalities per 100k population (Census PEP) ..."):
        rate_table, avg_rate = fatality_rate_per_100k_by_year(
            fatal_df, int(state_code), int(from_year), int(to_year), census_key
        )

    # Q2: average age
    avg_age = None
    if "AGE" in fatal_df.columns and not fatal_df.empty:
        ages = fatal_df["AGE"].dropna()
        avg_age = float(ages.mean()) if len(ages) else None

    # Q3: alcohol % + drug %
    bac = pick_bac_series(fatal_df)
    alcohol_pct = None
    if bac is not None and len(bac):
        alcohol_pct = float((bac >= float(bac_threshold)).mean() * 100.0)

    drug_pct = drug_involvement_percent(fatal_df, drugs_df if "drugs_df" in locals() else None)

    st.subheader(f"Results for {state_label} ({int(from_year)}–{int(to_year)})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q1) Avg fatality rate (per 100k)", f"{avg_rate:.2f}")
    c2.metric("Q2) Avg age (fatalities)", f"{avg_age:.1f}" if avg_age is not None else "N/A")
    c3.metric("Q3) % alcohol-involved", f"{alcohol_pct:.1f}%" if alcohol_pct is not None else "N/A")
    c4.metric("Q3) % drug-involved", f"{drug_pct:.1f}%" if drug_pct is not None else "N/A")

    st.divider()
    st.markdown("### Q1 Detail — Fatalities per 100,000 population (by year)")
    st.dataframe(rate_table, use_container_width=True)

    with st.expander("Raw NHTSA PERSON (first 200 rows)"):
        st.dataframe(person.head(200), use_container_width=True)

    if "drugs_df" in locals() and drugs_df is not None:
        with st.expander("Raw NHTSA DRUGS (first 200 rows)"):
            st.dataframe(drugs_df.head(200), use_container_width=True)

    st.markdown("---")
    st.caption("© Abrivio • Sources: NHTSA FARS (CrashAPI) + U.S. Census PEP • Screening analytics only.")
