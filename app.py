import os
from datetime import datetime, date

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
import sys, os
import streamlit as st

st.write("Python:", sys.version)
st.write("CWD:", os.getcwd())
st.write("Files in CWD:", os.listdir("."))

try:
    import plotly
    st.write("plotly version:", plotly.__version__)
except Exception as e:
    st.error(f"plotly import failed: {type(e).__name__}: {e}")
    st.stop()

import plotly.express as px

from ai_enrichment import ai_enrich_contacts
from osint_core import run_scraping
from osint_agent_antibot_v3_2 import CATEGORIES
from utils import (
    create_excel_4sheets,
    create_csv_segments,
    generate_outreach_templates,
)

load_dotenv()

st.set_page_config(
    page_title="🧠 OSINT Agent Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 OSINT Agent Enterprise v3.2")
st.markdown("**Scraping → AI Enrichment → Lead Generation completa**")

# -----------------------------
# Sidebar - Configurazione
# -----------------------------
st.sidebar.header("⚙️ Configurazione")

api_provider = st.sidebar.selectbox("AI Provider", ["Claude (Anthropic)", "Groq"])
api_key_env = os.getenv("ANTHROPIC_API_KEY") if api_provider == "Claude (Anthropic)" else os.getenv("GROQ_API_KEY")

api_key = st.sidebar.text_input(
    "API Key",
    value=api_key_env or "",
    type="password",
    help="Puoi anche impostare ANTHROPIC_API_KEY o GROQ_API_KEY come variabili d'ambiente."
)

start_date = st.sidebar.date_input("Data inizio", date(2024, 1, 1))
end_date = st.sidebar.date_input("Data fine", datetime.now().date())

categorie = st.sidebar.multiselect(
    "Categorie",
    options=list(CATEGORIES.keys()),
    default=list(CATEGORIES.keys())[:3],
)

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("📂 CSV Portali (obbligatorio)", type="csv")

capoluoghi_df = None
if uploaded_file is not None:
    capoluoghi_df = pd.read_csv(uploaded_file)
    st.success(f"Caricato: {len(capoluoghi_df)} portali")

    preview_cols = [c for c in ["COMUNE", "PROVINCIA", "REGIONE", "ALBO_PRETORIO_URL"] if c in capoluoghi_df.columns]
    if preview_cols:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(capoluoghi_df[preview_cols].head(), use_container_width=True)
        with col2:
            if "REGIONE" in capoluoghi_df.columns:
                st.metric("Regioni nel CSV", capoluoghi_df["REGIONE"].nunique())
            else:
                st.info("Colonna REGIONE non trovata nel CSV.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Scraping", "📊 Dashboard", "🤖 AI Enrichment", "📈 Export & Outreach"])

with tab1:
    st.header("1) Scraping Portali")

    if st.button("🚀 Avvia Scraping", type="primary", use_container_width=True):
        if capoluoghi_df is None:
            st.error("Carica prima il CSV dei portali.")
            st.stop()

        required = {"ALBO_PRETORIO_URL", "COMUNE", "PROVINCIA", "REGIONE"}
        missing = [c for c in required if c not in capoluoghi_df.columns]
        if missing:
            st.error(f"Nel CSV mancano queste colonne: {', '.join(missing)}")
            st.stop()

        with st.spinner("Scraping in corso..."):
            # Nota: run_scraping non filtra per date in modo affidabile finché non si estrae data dal PDF.
            raw_data = run_scraping(
                capoluoghi_df.to_dict("records"),
                start_date=start_date,
                end_date=end_date,
                categorie=categorie,
            )
            st.session_state["raw_projects"] = raw_data

        st.success(f"Scraping completato! Record: {len(st.session_state['raw_projects'])}")

    if "raw_projects" in st.session_state:
        df_raw = pd.DataFrame(st.session_state["raw_projects"])
        st.dataframe(df_raw, height=420, use_container_width=True)

with tab2:
    st.header("2) Dashboard")

    if "raw_projects" not in st.session_state:
        st.info("Esegui prima lo scraping (tab 🚀 Scraping).")
    else:
        df = pd.DataFrame(st.session_state["raw_projects"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Totale record", len(df))
        col2.metric("PDF ok", int(df.get("error", pd.Series([None]*len(df))).isna().sum()))
        col3.metric("Regioni coperte", int(df["regione"].nunique()) if "regione" in df.columns else 0)
        col4.metric("Comuni coperti", int(df["comune"].nunique()) if "comune" in df.columns else 0)

        if "regione" in df.columns:
            vc = df["regione"].value_counts().reset_index()
            vc.columns = ["regione", "count"]
            fig = px.bar(vc, x="regione", y="count", title="Record per Regione")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("3) AI Enrichment Contatti")

    if st.button("🤖 Arricchisci con AI", use_container_width=True):
        if "raw_projects" not in st.session_state:
            st.error("Prima fai lo scraping (tab 🚀 Scraping).")
            st.stop()

        if not api_key:
            st.warning("Nessuna API key: farò enrichment 'basic' (solo normalizzazione + score a 0).")

        with st.spinner("Arricchimento in corso..."):
            enriched_data = ai_enrich_contacts(
                st.session_state["raw_projects"],
                api_key=api_key,
                provider=api_provider,
            )
            st.session_state["enriched_leads"] = enriched_data

        st.success(f"Enrichment completato! Record: {len(st.session_state['enriched_leads'])}")

    if "enriched_leads" in st.session_state:
        df_leads = pd.DataFrame(st.session_state["enriched_leads"])

        display_cols = [c for c in ["progettista_norm", "email", "telefono", "validation_score", "lead_quality_score", "comune", "regione"] if c in df_leads.columns]
        if display_cols:
            st.dataframe(df_leads[display_cols], height=520, use_container_width=True)
        else:
            st.dataframe(df_leads, height=520, use_container_width=True)

        if "lead_quality_score" in df_leads.columns:
            segment_a = df_leads[df_leads["lead_quality_score"] >= 8.5]
            st.metric("🟢 Segment A (Premium)", len(segment_a))

with tab4:
    st.header("4) Export & Outreach")

    if "enriched_leads" not in st.session_state:
        st.info("Esegui enrichment prima (tab 🤖 AI Enrichment).")
    else:
        leads = st.session_state["enriched_leads"]

        col1, col2, col3 = st.columns(3)

        with col1:
            excel_data = create_excel_4sheets(leads)
            st.download_button(
                "📊 Excel 4-sheet completo",
                data=excel_data,
                file_name=f"osint_leads_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col2:
            csv_data = create_csv_segments(leads)
            st.download_button(
                "📋 CSV segmentato (A/B/C/D)",
                data=csv_data,
                file_name="leads_segments.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col3:
            templates = generate_outreach_templates(leads[:10])
            st.download_button(
                "✉️ Email templates",
                data=templates,
                file_name="email_templates.txt",
                mime="text/plain",
                use_container_width=True,
            )
