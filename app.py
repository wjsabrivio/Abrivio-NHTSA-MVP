import streamlit as st
import pandas as pd
import requests
import io
import zipfile
from functools import lru_cache
from typing import Optional, Tuple

# ============================
# Abrivio | NHTSA Regional Safety Insights (MVP)
# Data sources:
#  - NHTSA FARS annual CSV ZIP distributions (preferred; avoids CrashAPI 403 on Streamlit Cloud)
#  - U.S. Census PEP population API for fatality rate denominator (best-effort; non-fatal if unavailable)
# ============================

st.set_page_config(page_title="Abrivio | NHTSA Regional Safety Insights", layout="wide")

CENSUS_BASE = "https://api.census.gov/data"
NHTSA_DOWNLOAD_REDIRECTOR = "https://www.nhtsa.gov/file-downloads/download"
NHTSA_STATIC_BASE = "https://static.nhtsa.gov/nhtsa/downloads/FARS"

# Browser-like headers help with some .gov/CDN gatekeeping
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AbrivioMVP/1.0)",
    "Accept": "*/*",
}

# FARS STATE codes (includes PR=72). Used for dropdown and as fallback when NHTSA definitions endpoint is blocked.
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
# Utilities
# ----------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def safe_get(url: str, params: Optional[dict] = None, timeout: int = 120) -> requests.Response:
    r = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=timeout)
    return r

# ----------------------------
# Census: Population Estimates (best-effort)
# ----------------------------
@lru_cache(maxsize=256)
def census_state_population(state_fips: int, year: int, census_api_key: Optional[str]) -> Optional[int]:
    """
    Best-effort Census PEP population for state+year.
    If it fails (including PR quirks), returns None (non-fatal).
    """
    try:
        url = f"{CENSUS_BASE}/{year}/pep/population"
        params = {"get": f"NAME,POP_{year}", "for": f"state:{state_fips:02d}"}
        if census_api_key:
            params["key"] = census_api_key

        r = safe_get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        header, row = data[0], data[1]
        pop_col = header.index(f"POP_{year}")
        return int(row[pop_col])
    except Exception:
        return None

# ----------------------------
# NHTSA FARS: Annual CSV ZIP downloads (preferred)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)  # cache 7 days
def download_fars_zip(year: int, folder: str, zip_name: str) -> bytes:
    """
    Downloads official NHTSA FARS ZIP file bytes for a given year/folder/zip_name.
    Tries:
      1) NHTSA file-download redirector
      2) static.nhtsa.gov direct
    """
    # 1) Redirector
    params1 = {"p": f"nhtsa/downloads/FARS/{year}/{folder}/{zip_name}"}
    r1 = safe_get(NHTSA_DOWNLOAD_REDIRECTOR, params=params1, timeout=120)
    if r1.status_code == 200 and len(r1.content) > 5000:
        return r1.content

    # 2) Static
    url2 = f"{NHTSA_STATIC_BASE}/{year}/{folder}/{zip_name}"
    r2 = safe_get(url2, params=None, timeout=120)
    if r2.status_code == 200 and len(r2.content) > 5000:
        return r2.content

    raise RuntimeError(f"Could not download FARS ZIP: {year}/{folder}/{zip_name} (HTTP {r2.status_code})")

def read_csv_from_zip(zip_bytes: bytes, target_filename: str, usecols: Optional[list] = None) -> pd.DataFrame:
    """
    Reads a CSV member from a ZIP where member name endswith target_filename (case-insensitive).
    """
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    match = None
    for name in z.namelist():
        if name.upper().endswith(target_filename.upper()):
            match = name
            break
    if match is None:
        raise RuntimeError(f"{target_filename} not found in ZIP. Members: {z.namelist()[:20]} ...")

    with z.open(match) as f:
        return pd.read_csv(f, low_memory=False, usecols=usecols)

def possible_zip_names(year: int, is_pr: bool) -> list:
    """
    NHTSA naming can vary; try a few common patterns.
    """
    if is_pr:
        return [
            f"FARS{year}PuertoRicoCSV.zip",
            f"FARS{year}Puerto-RicoCSV.zip",
            f"FARS{year}PuertoRico.zip",
            f"FARS{year}Puerto-Rico.zip",
        ]
    return [
        f"FARS{year}NationalCSV.zip",
        f"FARS{year}National.zip",
        f"FARS{year}NationalCSV.ZIP",
        f"FARS{year}National.ZIP",
    ]

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def load_fars_person_drugs(year: int, state_code: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Returns (PERSON, DRUGS) for a single year and selected state.

    - For Puerto Rico (state 72): tries PR folders and PR zip names.
    - For all other states: uses National folder + filters by STATE.
    """
    is_pr = (state_code == 72)

    if is_pr:
        folder_candidates = ["Puerto-Rico", "Puerto_Rico", "PuertoRico", "Puerto Rico"]
    else:
        folder_candidates = ["National"]

    last_err = None

    # Narrow columns early to save memory on Streamlit Cloud
    person_cols = None  # keep None to be robust across vintages; set if you want hard optimization later
    drugs_cols = None

    for folder in folder_candidates:
        for zip_name in possible_zip_names(year, is_pr=is_pr):
            try:
                zip_bytes = download_fars_zip(year, folder, zip_name)

                person = read_csv_from_zip(zip_bytes, "PERSON.CSV", usecols=person_cols)
                person.columns = [c.upper() for c in person.columns]

                # DRUGS is optional in some vintages
                try:
                    drugs = read_csv_from_zip(zip_bytes, "DRUGS.CSV", usecols=drugs_cols)
                    drugs.columns = [c.upper() for c in drugs.columns]
                except Exception:
                    drugs = None

                # Filter national by state if possible
                if not is_pr and "STATE" in person.columns:
                    person = person[to_num(person["STATE"]) == state_code].copy()
                if (drugs is not None) and (not is_pr) and ("STATE" in drugs.columns):
                    drugs = drugs[to_num(drugs["STATE"]) == state_code].copy()

                # Ensure YEAR exists
                if "YEAR" not in person.columns:
                    person["YEAR"] = year
                if drugs is not None and "YEAR" not in drugs.columns:
                    drugs["YEAR"] = year

                return person, drugs

            except Exception as e:
                last_err = e

    raise RuntimeError(f"Failed to load FARS PERSON/DRUGS for year={year}, state={state_code}. Last error: {last_err}")

# ----------------------------
# Analytics
# ----------------------------
def compute_fatalities(person_df: pd.DataFrame) -> pd.DataFrame:
    if person_df.empty:
        return person_df

    df = person_df.copy()
    df.columns = [c.upper() for c in df.columns]

    df["INJ_SEV"] = to_num(df.get("INJ_SEV"))

    # Fatal injury commonly corresponds to INJ_SEV == 4
    fatal = df[df["INJ_SEV"] == 4].copy()

    if "YEAR" in fatal.columns:
        fatal["YEAR"] = to_num(fatal["YEAR"])
    if "AGE" in fatal.columns:
        fatal["AGE"] = to_num(fatal["AGE"])

    return fatal

def pick_bac_series(fatal_df: pd.DataFrame) -> Optional[pd.Series]:
    for c in ["BAC", "ALC_RES", "ALCRES", "ALCDET"]:
        if c in fatal_df.columns:
            return to_num(fatal_df[c])
    return None

def drug_involvement_percent(fatal_df: pd.DataFrame, drugs_df: Optional[pd.DataFrame]) -> Optional[float]:
    if drugs_df is None or drugs_df.empty or fatal_df.empty:
        return None

    f = fatal_df.copy()
    d = drugs_df.copy()
    f.columns = [c.upper() for c in f.columns]
    d.columns = [c.upper() for c in d.columns]

    keys = [k for k in ["ST_CASE", "VEH_NO", "PER_NO"] if k in f.columns and k in d.columns]
    if len(keys) < 2:
        return None

    m = f.merge(d, on=keys, how="left", suffixes=("", "_DRUG"))

    drug_cols = [c for c in m.columns if "DRUG" in c]
    if not drug_cols:
        return None

    has_drug = m[drug_cols].notna().any(axis=1)
    return float(has_drug.mean() * 100.0)

def fatality_rate_per_100k_by_year(
    fatal_df: pd.DataFrame,
    state_fips: int,
    from_year: int,
    to_year: int,
    census_api_key: Optional[str],
) -> Tuple[pd.DataFrame, Optional[float]]:

    rows = []
    for y in range(from_year, to_year + 1):
        deaths = int((fatal_df.get("YEAR", pd.Series([], dtype=float)) == y).sum()) if "YEAR" in fatal_df.columns else int(len(fatal_df))

        pop = census_state_population(state_fips, y, census_api_key)
        if pop is None or pop == 0:
            rate = None
        else:
            rate = (deaths / pop) * 100000.0

        rows.append({"year": y, "fatalities": deaths, "population": pop, "rate_per_100k": rate})

    table = pd.DataFrame(rows)

    # Compute mean ignoring missing rates
    if table["rate_per_100k"].notna().any():
        avg_rate = float(table["rate_per_100k"].dropna().mean())
    else:
        avg_rate = None

    return table, avg_rate

# ----------------------------
# Demo Mode
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
st.caption("MVP: NHTSA FARS annual CSV files + Census Population Estimates (best-effort) to answer 3 regional safety questions.")

with st.sidebar:
    st.header("Controls")
    demo_mode = st.toggle("Demo mode (no internet)", value=False)

    # Keep this OFF by default on Streamlit Cloud; it often 403s.
    try_definitions = st.toggle("Try NHTSA definitions endpoint (may be blocked)", value=False, disabled=demo_mode)

    if not demo_mode:
        if try_definitions:
            # Optional: Try NHTSA CrashAPI definitions; if blocked, fallback
            # (We intentionally avoid importing/using CrashAPI for data pulls.)
            NHTSA_BASE = "https://crashviewer.nhtsa.dot.gov/CrashAPI"

            @lru_cache(maxsize=16)
            def nhtsa_variable_attributes(variable: str, case_year: int) -> pd.DataFrame:
                url = f"{NHTSA_BASE}/definitions/GetVariableAttributes"
                params = {"variable": variable, "caseYear": case_year, "format": "json"}
                r = safe_get(url, params=params, timeout=60)
                r.raise_for_status()
                j = r.json()
                return pd.DataFrame(j.get("Results", []))

            defs_year = st.number_input("NHTSA definitions year", min_value=2010, max_value=2024, value=2023, step=1)

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
                st.warning("NHTSA definitions endpoint blocked. Using built-in FARS state list.")
                state_label = st.selectbox("Region (State)", sorted(FARS_STATE_CODES.keys()))
                state_code = int(FARS_STATE_CODES[state_label])
        else:
            # Default: built-in valid list (fast + reliable)
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

# Streamlit Secrets (optional)
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

        # Override population function in demo mode
        @lru_cache(maxsize=256)
        def demo_pop(state_fips, year, key=None):  # noqa: ARG001
            return pop.get(int(year), None)

        globals()["census_state_population"] = demo_pop

        person = fatal
        drugs_df = drugs
    else:
        # TIP: start with a smaller year range if you want faster first load
        with st.spinner("Downloading official FARS annual CSV files (cached) ..."):
            person_list = []
            drugs_list = []

            for y in range(int(from_year), int(to_year) + 1):
                p, d = load_fars_person_drugs(y, int(state_code))
                person_list.append(p)
                if d is not None:
                    drugs_list.append(d)

            person = pd.concat(person_list, ignore_index=True) if person_list else pd.DataFrame()
            drugs_df = pd.concat(drugs_list, ignore_index=True) if drugs_list else None

    fatal_df = compute_fatalities(person)

    with st.spinner("Computing fatalities per 100k population (Census PEP; best-effort) ..."):
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

    drug_pct = drug_involvement_percent(fatal_df, drugs_df)

    st.subheader(f"Results for {state_label} ({int(from_year)}–{int(to_year)})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q1) Avg fatality rate (per 100k)", f"{avg_rate:.2f}" if avg_rate is not None else "N/A")
    c2.metric("Q2) Avg age (fatalities)", f"{avg_age:.1f}" if avg_age is not None else "N/A")
    c3.metric("Q3) % alcohol-involved", f"{alcohol_pct:.1f}%" if alcohol_pct is not None else "N/A")
    c4.metric("Q3) % drug-involved", f"{drug_pct:.1f}%" if drug_pct is not None else "N/A")

    st.divider()
    st.markdown("### Q1 Detail — Fatalities per 100,000 population (by year)")
    st.dataframe(rate_table, use_container_width=True)

    with st.expander("Raw NHTSA FARS PERSON (first 200 rows)"):
        st.dataframe(person.head(200), use_container_width=True)

    if drugs_df is not None:
        with st.expander("Raw NHTSA FARS DRUGS (first 200 rows)"):
            st.dataframe(drugs_df.head(200), use_container_width=True)

    st.markdown("---")
    st.caption("© Abrivio • Sources: NHTSA FARS annual CSV files + U.S. Census PEP (best-effort) • Screening analytics only.")
