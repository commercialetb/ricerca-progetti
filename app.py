import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import os
import json
from datetime import datetime
import requests
from io import BytesIO
import re
import sys

# Integra il TUO scraper [file:1]
sys.path.append(".")
from osint_agent_antibot_v3_2 import BrowserPool, SeleniumCrawler, CATEGORIES  # [file:1]

st.set_page_config(page_title="🧠 OSINT Agent con MENU REGIONI", layout="wide")

st.title("🧠 OSINT Agent ENTERPRISE")
st.markdown("**Menu Regioni + Segmenti + Scraping + AI**")

# ========== MENU REGIONI E SEGMENTI (PRIMO LIVELLO) ==========
col_menu1, col_menu2 = st.columns([1, 3])

with col_menu1:
    st.header("🎯 SELEZIONI")
    
    # MENU REGIONI COMPLETE [web:19][web:21]
    REGIONI_ITALIA = [
        "Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna",
        "Friuli-Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche",
        "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia", "Toscana",
        "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto"
    ]
    
    seleziona_tutto_regioni = st.checkbox("✅ **Tutte le Regioni**", value=True)
    if not seleziona_tutto_regioni:
        regioni_selezionate = st.multiselect(
            "Scegli regioni:",
            REGIONI_ITALIA,
            default=REGIONI_ITALIA[:3]  # Prime 3 di default
        )
    else:
        regioni_selezionate = REGIONI_ITALIA
    
    # MENU SEGMENTI OUTREACH
    SEGMENTI = ["A Premium (8.5-10)", "B Good (7-8.4)", "C Warm (5-6.9)", "D Follow-up (<5)", "Tutti"]
    segmento_selezionato = st.selectbox("Segmento Outreach:", SEGMENTI, index=4)
    
    # CATEGORIE PROGETTI
    tutte_categorie = st.checkbox("🏗️ Tutte le categorie", value=True)
    if not tutte_categorie:
        categorie_sel = st.multiselect("Categorie:", list(CATEGORIES.keys()), default=list(CATEGORIES.keys())[:3])

with col_menu2:
    st.header("📁 UPLOAD MASTER PORTALI")
    uploaded_csv = st.file_uploader("**CSV Portali (MASTER_SA) 👇**", type="csv")
    
    if uploaded_csv is not None:
        master_df = pd.read_csv(uploaded_csv)
        # FILTRA per REGIONI selezionate
        master_filtrato = master_df[master_df["REGIONE"].isin(regioni_selezionate)]
        st.success(f"✅ {len(master_filtrato)} portali per {len(regioni_selezionate)} regioni")
        st.dataframe(master_filtrato[["COMUNE", "REGIONE", "ALBO_PRETORIO_URL"]].head())
        st.session_state["master_filtrato"] = master_filtrato
    else:
        st.warning("Carica CSV per procedere")

# ========== TAB WORKFLOW ==========
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Scraping", "2️⃣ AI Enrichment", "3️⃣ Dashboard", "4️⃣ Export"])

# TAB 1: SCRAPING FILTRATO PER REGIONE
with tab1:
    st.header("🔍 Scraping Selezionato")
    if "master_filtrato" not in st.session_state:
        st.warning("👆 Seleziona regioni e carica CSV")
        st.stop()
    
    max_portali = st.slider(f"Portali da {', '.join(regioni_selezionate[:2])}...", 1, 50, 10)
    
    if st.button(f"🚀 Scraping {len(regioni_selezionate)} Regioni", type="primary"):
        with st.spinner("Anti-bot scraping..."):
            results = run_regione_scraping(st.session_state["master_filtrato"].head(max_portali))
            results["regioni_target"] = ", ".join(regioni_selezionate)
            results.to_csv("data/scraping_regioni.csv", index=False)
            st.session_state["scraped_regioni"] = results
            st.success(f"✅ {len(results)} progetti scrapati!")

# TAB 2: AI PER SEGMENTO SELEZIONATO
with tab2:
    st.header("🎯 AI per Segmento")
    if "scraped_regioni" not in st.session_state:
        st.warning("👆 Scraping prima")
        st.stop()
    
    claude_key = st.text_input("Claude API Key", type="password")
    if claude_key and st.button(f"🤖 AI Enrichment per {segmento_selezionato}"):
        df_ai = ai_enrich_segment(st.session_state["scraped_regioni"], claude_key, segmento_selezionato)
        df_ai.to_csv("data/ai_segmento.csv", index=False)
        st.session_state["ai_results"] = df_ai
        st.success("✅ AI completato!")

# TAB 3: DASHBOARD CON FILTRO SEGMENTO
with tab3:
    st.header("📊 Dashboard Filtrata")
    df_dash = st.session_state.get("ai_results", st.session_state.get("scraped_regioni", pd.DataFrame()))
    
    # KPI per segmento
    if "score" in df_dash.columns:
        segment_data = df_dash["score"].describe()
        c1, c2, c3 = st.columns(3)
        c1.metric("Leads Totali", len(df_dash))
        c2.metric(f"{segmento_selezionato}", len(df_dash[df_dash["score"] >= 8.5 if 'Premium' in segmento_selezionato else 0]))
        c3.metric("Media Score", f"{segment_data['mean']:.1f}")
    
    # Grafico regioni selezionate
    fig = px.bar(
        df_dash.groupby("regione").size().reset_index(name="progetti"),
        x="regione", y="progetti",
        title=f"Progetti nelle {len(regioni_selezionate)} regioni selezionate",
        color="progetti"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabella filtro segmento
    if "score" in df_dash.columns:
        if "Premium" in segmento_selezionato:
            df_show = df_dash[df_dash["score"] >= 8.5]
        elif "Tutti" not in segmento_selezionato:
            soglia = {"A": 8.5, "B": 7.0, "C": 5.0, "D": 0}.get(segmento_selezionato[0], 0)
            df_show = df_dash[(df_dash["score"] >= soglia) & (df_dash["score"] < soglia+1.5)]
        else:
            df_show = df_dash
        
        st.dataframe(df_show[["comune", "regione", "email", "score", "portal_url"]])

# TAB 4: EXPORT PER SEGMENTO
with tab4:
    st.header("💾 Export Segmento")
    if "ai_results" in st.session_state:
        df_export = st.session_state["ai_results"]
        
        # CSV specifico segmento
        if segmento_selezionato != "Tutti":
            soglia = {"A": 8.5, "B": 7.0, "C": 5.0, "D": 0}[segmento_selezionato[0]]
            df_seg = df_export[df_export["score"] >= soglia]
            csv_seg = df_seg.to_csv(index=False).encode()
            st.download_button(
                f"📥 {segmento_selezionato}",
                csv_seg,
                f"leads_{segmento_selezionato}_{'_'.join(regioni_selezionate[:2])}.csv"
            )
        
        # Excel completo
        excel_file = create_excel_final(df_export, regioni_selezionate, segmento_selezionato)
        st.download_button("📊 Excel 4-Sheet", excel_file)

# ========== FUNZIONI UTILITY ==========
def run_regione_scraping(portali_df):
    """Scraping solo regioni selezionate"""
    browser_pool = BrowserPool(pool_size=2)
    crawler = SeleniumCrawler(browser_pool)
    results = []
    for _, portal in portali_df.iterrows():
        pdfs = crawler.get_pdf_links(portal["ALBO_PRETORIO_URL"])
        results.append({
            "comune": portal["COMUNE"],
            "regione": portal["REGIONE"],
            "portal_url": portal["ALBO_PRETORIO_URL"],
            "n_pdf": len(pdfs)
        })
    browser_pool.cleanup()
    return pd.DataFrame(results)

def ai_enrich_segment(df_scraped, api_key, segmento):
    """AI filtrata per segmento"""
    # Mock AI (sostituisci con Claude reale)
    df_enriched = df_scraped.copy()
    df_enriched["score"] = pd.Series(range(5,11)).sample(len(df_scraped)).values  # Simulazione
    df_enriched["email"] = ["info@studio" + str(i) + ".it" for i in range(len(df_scraped))]
    return df_enriched

def create_excel_final(df, regioni, segmento):
    """Excel 4 sheet"""
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        df.to_excel(writer, "ALL_LEADS")
        pd.DataFrame(df.groupby("regione").size()).to_excel(writer, "BY_REGIONE")
    return output.getvalue()

st.markdown("---")
st.caption(f"**Regioni selezionate**: {', '.join(regioni_selezionate)} | **Segmento**: {segmento_selezionato}")
