# app.py - Streamlit UI
from __future__ import annotations

import os
from typing import List

import pandas as pd
import streamlit as st

from osint_core import CATEGORIES, run_scraping
from ai_enrichment import ai_enrich_contacts
from utils import (
    create_csv_segments,
    create_excel_4sheets,
    generate_outreach_templates,
    normalize_progettista,
)

# Plotly è opzionale: su Streamlit Cloud può mancare / fallire su py3.13
try:
    import plotly.express as px  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


st.set_page_config(page_title="Ricerca Progetti (OSINT)", layout="wide")

st.title("Ricerca Progetti Italia – OSINT Agent")
st.caption("Versione cloud robusta: fallback senza Selenium/OCR, nessun crash se mancano dipendenze")

# --- Sidebar
st.sidebar.header("Impostazioni")

# Carica master CSV (se presente nel repo) o upload
default_master_path = "MASTER_SA_gare_links_NORMALIZED.csv"
master_df = None

if os.path.exists(default_master_path):
    try:
        master_df = pd.read_csv(default_master_path)
        st.sidebar.success(f"Caricato: {default_master_path} ({len(master_df)} righe)")
    except Exception as e:
        st.sidebar.warning(f"Impossibile leggere {default_master_path}: {e}")

uploaded = st.sidebar.file_uploader("Oppure carica CSV portali", type=["csv"])
if uploaded is not None:
    try:
        master_df = pd.read_csv(uploaded)
        st.sidebar.success(f"Caricato upload ({len(master_df)} righe)")
    except Exception as e:
        st.sidebar.error(f"CSV non valido: {e}")

if master_df is None:
    st.info("Carica un CSV (con colonna ALBO_PRETORIO_URL o PORTAL_URL) per iniziare.")
    st.stop()

# Filtri
regioni = sorted(
    {
        str(x).strip()
        for x in master_df.get("REGIONE", pd.Series(dtype=str)).dropna().tolist()
        if str(x).strip()
    }
)
regione_sel = st.sidebar.selectbox("Regione", ["(tutte)"] + regioni)

if regione_sel != "(tutte)" and "REGIONE" in master_df.columns:
    df_portals = master_df[master_df["REGIONE"].astype(str).str.strip() == regione_sel].copy()
else:
    df_portals = master_df.copy()

categories_sel = st.sidebar.multiselect(
    "Categorie (keyword pack)",
    options=list(CATEGORIES.keys()),
    default=list(CATEGORIES.keys()),
)
max_pages = st.sidebar.slider("Profondità (pagine)", 1, 10, 3)

st.sidebar.divider()
st.sidebar.subheader("AI enrichment (opzionale)")
provider = st.sidebar.selectbox("Provider", ["anthropic", "groq"], index=0)
api_key = st.sidebar.text_input("API key", type="password", help="Se vuota, l'enrichment viene saltato")

# --- Main
colA, colB = st.columns([2, 1])
with colA:
    st.subheader("Portali selezionati")
    st.dataframe(df_portals.head(200), use_container_width=True)
with colB:
    st.subheader("Statistiche")
    st.metric("Portali", len(df_portals))
    st.metric("Categorie", len(categories_sel))

run_btn = st.button("🚀 Avvia ricerca", type="primary")

if run_btn:
    with st.spinner("Scraping in corso..."):
        projects = run_scraping(df_portals, categories=categories_sel, max_pages=max_pages)

    if not projects:
        st.warning("Nessun PDF trovato (o portali non raggiungibili). Prova ad aumentare max_pages o cambia regione.")
        st.stop()

    st.success(f"Trovati {len(projects)} PDF/progetti (grezzi)")

    # Normalizza progettista
    for p in projects:
        if "progettista_norm" not in p:
            p["progettista_norm"] = normalize_progettista(p.get("progettista_raw", ""))

    df_projects = pd.DataFrame(projects)

    st.subheader("Risultati grezzi")
    st.dataframe(df_projects, use_container_width=True)

    # AI enrichment
    with st.spinner("Enrichment (opzionale)..."):
        enriched = ai_enrich_contacts(projects, api_key=api_key, provider=provider)

    df_enriched = pd.DataFrame(enriched)
    st.subheader("Risultati arricchiti")
    st.dataframe(df_enriched, use_container_width=True)

    # Piccola viz: per regione
    if "regione" in df_enriched.columns:
        counts = df_enriched["regione"].fillna("(vuoto)").value_counts().reset_index()
        counts.columns = ["regione", "n"]
        st.subheader("Distribuzione per regione")
        if PLOTLY_AVAILABLE:
            fig = px.bar(counts, x="regione", y="n")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(counts.set_index("regione"))

    # Export
    st.subheader("Export")
    xlsx_bytes = create_excel_4sheets(enriched)
    csv_bytes = create_csv_segments(enriched)

    st.download_button(
        "⬇️ Scarica Excel (4 sheet)",
        data=xlsx_bytes,
        file_name="osint_leads.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "⬇️ Scarica CSV (segmentato)",
        data=csv_bytes,
        file_name="osint_leads.csv",
        mime="text/csv",
    )

    # Template outreach
    st.subheader("Template outreach")
    top = (
        df_enriched.sort_values(by="lead_quality_score", ascending=False)
        .head(15)
        .to_dict(orient="records")
        if not df_enriched.empty
        else []
    )
    templates_txt = generate_outreach_templates(top)
    st.text_area("Email templates", templates_txt, height=260)
    st.download_button(
        "⬇️ Scarica template email (txt)",
        data=templates_txt.encode("utf-8"),
        file_name="email_templates.txt",
        mime="text/plain",
    )
