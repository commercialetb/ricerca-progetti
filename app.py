import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="🧠 OSINT Agent Enterprise",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 OSINT Agent Enterprise v3.2")
st.markdown("**Scraping → AI Enrichment → Lead Generation completa**")

# Sidebar - Configurazione
st.sidebar.header("⚙️ Configurazione")

# API Keys
api_provider = st.sidebar.selectbox("AI Provider", ["Claude (Anthropic)", "Groq"])
if api_provider == "Claude (Anthropic)":
    api_key = st.sidebar.text_input("Anthropic API Key", type="password")
else:
    api_key = st.sidebar.text_input("Groq API Key", type="password")

start_date = st.sidebar.date_input("Data inizio", datetime(2024, 1, 1))
end_date = st.sidebar.date_input("Data fine", datetime.now())
categorie = st.sidebar.multiselect("Categorie", list(CATEGORIES.keys()), default=list(CATEGORIES.keys())[:3])

# Upload CSV
uploaded_file = st.file_uploader("📂 CSV Portali (obbligatorio)", type="csv")
if uploaded_file:
    capoluoghi_df = pd.read_csv(uploaded_file)
    st.success(f"Caricato: {len(capoluoghi_df)} portali")
    
    # Anteprima
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(capoluoghi_df[["COMUNE", "PROVINCIA", "REGIONE", "ALBO_PRETORIO_URL"]].head())
    with col2:
        st.metric("Portali per Regione", capoluoghi_df["REGIONE"].value_counts().head(1))

# Tab principali
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Scraping", "📊 Dashboard", "🤖 AI Enrichment", "📈 Export & Outreach"])

with tab1:
    st.header("1. Scraping Portali")
    
    if st.button("🚀 Avvia Scraping", type="primary", use_container_width=True):
        with st.spinner("Scraping in corso..."):
            raw_data = run_scraping(capoluoghi_df.to_dict("records"), start_date, end_date)
            st.session_state["raw_projects"] = raw_data
            st.success("Scraping completato!")
    
    if "raw_projects" in st.session_state:
        df_raw = pd.DataFrame(st.session_state["raw_projects"])
        st.dataframe(df_raw, height=400)

with tab2:
    st.header("2. Dashboard Regionale")
    if "raw_projects" in st.session_state:
        df_dashboard = pd.DataFrame(st.session_state["raw_projects"])
        
        # Map regionale
        fig = px.scatter_geo(
            df_dashboard, 
            lat="lat", lon="lon",
            hover_name="COMUNE",
            size="n_pdf",
            color="REGIONE",
            title="Progetti per Regione"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metriche
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Totale PDF", len(df_dashboard))
        col2.metric("Portali OK", len(df_dashboard[df_dashboard["error"].isna()]))
        col3.metric("Regioni coperte", df_dashboard["REGIONE"].nunique())
        col4.metric("Media PDF/portale", df_dashboard["n_pdf"].mean())

with tab3:
    st.header("3. AI Enrichment Contatti")
    if st.button("🤖 Arricchisci con AI") and "raw_projects" in st.session_state:
        with st.spinner("Ricerca contatti in corso..."):
            enriched_data = ai_enrich_contacts(
                st.session_state["raw_projects"], 
                api_key=api_key, 
                provider=api_provider
            )
            st.session_state["enriched_leads"] = enriched_data
            st.success("Enrichment completato!")
    
    if "enriched_leads" in st.session_state:
        df_leads = pd.DataFrame(st.session_state["enriched_leads"])
        
        # Tabella leads con filtri
        st.dataframe(df_leads[["PROGETTISTA", "EMAIL", "TELEFONO", "VALIDATION_SCORE", "LEAD_SCORE"]], height=500)
        
        # Segmenti A/B/C/D
        segment_a = df_leads[df_leads["LEAD_SCORE"] >= 8.5]
        st.metric("🟢 Segment A (Premium)", len(segment_a))

with tab4:
    st.header("4. Export & Outreach")
    
    if "enriched_leads" in st.session_state:
        # Download multipli
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Excel 4-sheet
            excel_data = create_excel_4sheets(st.session_state["enriched_leads"])
            st.download_button(
                "📊 Excel 4-sheet completo",
                data=excel_data,
                file_name=f"osint_leads_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # CSV Segmenti
            csv_segments = create_csv_segments(st.session_state["enriched_leads"])
            st.download_button(
                "📋 CSV A/B/C/D",
                data=csv_segments,
                file_name="leads_segments.csv",
                mime="text/csv"
            )
        
        with col3:
            # Email templates
            templates = generate_outreach_templates(st.session_state["enriched_leads"][:10])
            st.download_button(
                "✉️ Email Templates personalizzati",
                data=templates,
                file_name="email_templates.txt"
            )
