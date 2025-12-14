# Abrivio | NHTSA Regional Safety Insights

Streamlit MVP that pulls:
- NHTSA FARS (CrashAPI) PERSON (+ DRUGS best-effort)
- U.S. Census Population Estimates (PEP)

Answers:
1) Avg fatality rate per 100,000 population
2) Avg age of fatalities
3) % alcohol-involved and % drug-involved (best-effort)

## Run locally
pip install -r requirements.txt
streamlit run app.py
