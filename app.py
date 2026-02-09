import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_enrichment import ai_enrich_contacts
from osint_agent_antibot_v3_2 import CATEGORIES
from osint_core import run_scraping_light, run_scraping_selenium
from utils import create_csv_segments, create_excel_4sheets

load_dotenv()

st.set_page_config(
    page_title="🧠 OSINT Agent Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 OSINT Agent Enterprise (Cloud-safe)")
st.caption(
    "Modalità leggera di default (no Selenium/OCR). Abilita funzioni pesanti solo se hai installato le dipendenze."
)

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Configurazione")

feature_plotly = st.sidebar.toggle(
    "📊 Dashboard Plotly",
    value=False,
    help="Richiede plotly installato (è in requirements).",
)
feature_selenium = st.sidebar.toggle(
    "🧭 Scraping Selenium (anti-bot)",
    value=False,
    help="Richiede selenium + chromium + chromium-driver.",
)
feature_ai = st.sidebar.toggle(
    "🤖 AI Enrichment",
    value=False,
    help="Richiede chiave API + provider.",
)
feature_ocr = st.sidebar.toggle(
    "🖼 OCR PDF (lento)",
    value=False,
    help="Richiede pdf2image + pytesseract + poppler + tesseract.",
)

st.sidebar.divider()

api_provider = st.sidebar.selectbox(
    "AI Provider", ["Claude (Anthropic)", "Groq"], disabled=not feature_ai
)
api_key = st.sidebar.text_input("API Key", type="password", disabled=not feature_ai)

max_pages = st.sidebar.slider(
    "Pagine da scansionare per portale (light)", 1, 10, 5, disabled=feature_selenium
)
max_pdf = st.sidebar.slider("PDF max per portale", 5, 60, 20)

st.sidebar.info(
    "⚡ Per avvio rapido su Streamlit Cloud: lascia disattivato Selenium/OCR. "
    "Se li attivi senza dipendenze di sistema, vedrai un errore chiaro."
)

# ---------------- Upload CSV ----------------
uploaded_file = st.file_uploader("📂 CSV Portali (obbligatorio)", type="csv")

capoluoghi_df = None
if uploaded_file:
    capoluoghi_df = pd.read_csv(uploaded_file)
    st.success(f"Caricato: {len(capoluoghi_df)} portali")

    preview_cols = [
        c
        for c in ["COMUNE", "PROVINCIA", "REGIONE", "ALBO_PRETORIO_URL"]
        if c in capoluoghi_df.columns
    ]
    if preview_cols:
        st.dataframe(capoluoghi_df[preview_cols].head(20), use_container_width=True)
else:
    st.warning("Carica un CSV per iniziare.")

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Scraping", "📊 Dashboard", "🤖 AI Enrichment", "📈 Export"])

# ---------------- Tab 1: Scraping ----------------
with tab1:
    st.subheader("1) Scraping Portali")

    if capoluoghi_df is None:
        st.info("Carica il CSV dei portali.")
    else:
        colA, colB = st.columns([2, 1])
        with colA:
            if st.button("🚀 Avvia Scraping", type="primary", use_container_width=True):
                with st.spinner("Scraping in corso..."):
                    portals = capoluoghi_df.to_dict("records")

                    try:
                        if feature_selenium:
                            raw = run_scraping_selenium(portals, max_pdf_per_portale=max_pdf)
                        else:
                            raw = run_scraping_light(
                                portals,
                                max_pdf_per_portale=max_pdf,
                                max_pages_per_portale=max_pages,
                            )

                        st.session_state["raw_projects"] = raw
                        st.success(f"Scraping completato! Record: {len(raw)}")
                    except Exception as e:
                        st.error(f"Errore scraping: {e}")

        with colB:
            st.metric("Portali nel CSV", len(capoluoghi_df))

        if "raw_projects" in st.session_state:
            df_raw = pd.DataFrame(st.session_state["raw_projects"])
            st.dataframe(df_raw, height=450, use_container_width=True)

# ---------------- Tab 2: Dashboard ----------------
with tab2:
    st.subheader("2) Dashboard")
    if "raw_projects" not in st.session_state:
        st.info("Esegui lo scraping per vedere la dashboard.")
    else:
        df = pd.DataFrame(st.session_state["raw_projects"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Record", len(df))
        col2.metric(
            "PDF OK",
            int((~df.get("error", pd.Series([None] * len(df))).notna()).sum()) if len(df) else 0,
        )
        col3.metric("Regioni", df["regione"].nunique() if "regione" in df.columns else 0)

        if feature_plotly:
            try:
                import plotly.express as px  # lazy import

                if "regione" in df.columns:
                    chart = df["regione"].value_counts().reset_index()
                    chart.columns = ["regione", "count"]
                    fig = px.bar(chart, x="regione", y="count", title="Record per Regione")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Colonna 'regione' non trovata nei dati.")
            except Exception as e:
                st.error(f"Plotly non disponibile o errore: {e}")
        else:
            st.info("Attiva 'Dashboard Plotly' nella sidebar per i grafici.")

# ---------------- Tab 3: AI Enrichment ----------------
with tab3:
    st.subheader("3) AI Enrichment contatti")
    if "raw_projects" not in st.session_state:
        st.info("Esegui lo scraping prima.")
    else:
        if not feature_ai:
            st.info("Attiva 'AI Enrichment' nella sidebar (e inserisci API key).")
        else:
            if st.button("🤖 Arricchisci con AI", use_container_width=True):
                with st.spinner("Arricchimento in corso..."):
                    provider = "Claude (Anthropic)" if "claude" in api_provider.lower() else "Groq"
                    enriched = ai_enrich_contacts(
                        st.session_state["raw_projects"], api_key=api_key, provider=provider
                    )
                    st.session_state["enriched_leads"] = enriched
                    st.success("Enrichment completato!")

        if "enriched_leads" in st.session_state:
            df_leads = pd.DataFrame(st.session_state["enriched_leads"])
            show_cols = [
                c
                for c in [
                    "progettista_norm",
                    "email",
                    "telefono",
                    "validation_score",
                    "lead_quality_score",
                    "error_ai",
                    "pdf_source",
                ]
                if c in df_leads.columns
            ]
            st.dataframe(df_leads[show_cols], height=500, use_container_width=True)

# ---------------- Tab 4: Export ----------------
with tab4:
    st.subheader("4) Export")

    if "enriched_leads" not in st.session_state and "raw_projects" not in st.session_state:
        st.info("Esegui scraping (e opzionalmente enrichment) per esportare.")
    else:
        data_for_export = (
            st.session_state.get("enriched_leads")
            or st.session_state.get("raw_projects")
            or []
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            xlsx = create_excel_4sheets(data_for_export)
            st.download_button(
                "📊 Scarica Excel (4 sheet)",
                data=xlsx,
                file_name=f"osint_export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col2:
            combined_csv, per = create_csv_segments(data_for_export)
            st.download_button(
                "📋 Scarica CSV (tutti)",
                data=combined_csv,
                file_name="leads_all.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col3:
            st.download_button(
                "📋 Scarica CSV Segment A",
                data=per.get("A", ""),
                file_name="segment_A_premium.csv",
                mime="text/csv",
                use_container_width=True,
            )
