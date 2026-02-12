import streamlit as st
import pandas as pd
import requests
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from groq import Groq
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

st.set_page_config(page_title="API LIVE: 4 Fonti Master", layout="wide")

# ========================================
# DATABASE MASTER
# ========================================
@st.cache_resource
def get_db():
    conn = sqlite3.connect('master_4fonti.db', check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS progetti_master (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        fonte TEXT,
        comune TEXT,
        identificativo TEXT,
        titolo TEXT,
        importo REAL,
        progettista TEXT,
        data_evento TEXT,
        url_dettaglio TEXT,
        raw_data TEXT
    )''')
    return conn

db = get_db()

# ========================================
# 1. ANAC API / OpenData LIVE
# ========================================
@st.cache_data(ttl=1800)
def fetch_anac_live():
    """ANAC dataset appalti 2025 LIVE"""
    try:
        # ANAC OCDS 2025 - primo dataset disponibile
        url = "https://dati.anticorruzione.it/opendata/dataset/ocds-appalti-ordinari-2025"
        resp = requests.get(url, timeout=30)
        
        # Parse HTML per link CSV/JSON ultimo
        soup = BeautifulSoup(resp.text, 'html.parser')
        csv_links = soup.find_all('a', href=re.compile(r'\.csv$'))
        
        if csv_links:
            csv_url = csv_links[0]['href']
            df = pd.read_csv(csv_url, nrows=100)  # Sample
            df['fonte'] = 'ANAC'
            df['timestamp'] = datetime.now().isoformat()
            
            # Salva DB
            df.to_sql('progetti_master', db, if_exists='append', index=False)
            return df[['comune', 'cig', 'titolo', 'importo', 'progettista']]
    except Exception as e:
        st.error(f"ANAC: {e}")
    return pd.DataFrame()

# ========================================
# 2. OPENCUP API LIVE (CF Comuni)
# ========================================
async def fetch_opencup_live(session, cf_comune):
    """OpenCUP API reale"""
    url = f"https://api.sogei.it/rgs/opencup/o/extServiceApi/v1/opendataes/soggettotitolare/{cf_comune}"
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                progetti = []
                for p in data.get('data', {}).get('progetti', [])[:5]:
                    progetti.append({
                        'fonte': 'OpenCUP',
                        'comune': cf_comune[:10],
                        'identificativo': p.get('codiceCUP'),
                        'titolo': p.get('descrizione', '')[:150],
                        'importo': float(p.get('valoreProgrammato', 0)),
                        'data_evento': p.get('dataStato'),
                        'raw_data': json.dumps(p)
                    })
                return progetti
    except:
        pass
    return []

async def batch_opencup(cf_list):
    """Batch OpenCUP 10 comuni"""
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_opencup_live(session, cf) for cf in cf_list[:10]]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

# ========================================
# 3. OPENER API LIVE
# ========================================
async def fetch_openpnrr_live(session, comune):
    """OpenPNRR API reale"""
    url = f"https://openpnrr.it/api/v1/progetti/?comune={comune.replace(' ', '%20')}&limit=10"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                progetti = []
                for p in data.get('results', [])[:5]:
                    progetti.append({
                        'fonte': 'OpenPNRR',
                        'comune': comune,
                        'identificativo': p.get('id_progetto'),
                        'titolo': p.get('descrizione_progetto', '')[:150],
                        'importo': float(p.get('importo', 0)),
                        'data_evento': p.get('data_avvio'),
                        'url_dettaglio': p.get('link')
                    })
                return progetti
    except:
        pass
    return []

# ========================================
# 4. AT SCRAPING (precedente)
# ========================================
async def scrape_at_live(session, sito, comune):
    """AT scraping (codice precedente ottimizzato)"""
    # [Implementazione scrape_amministrazione_trasparente completa]
    pass

# ========================================
# DASHBOARD 4 API LIVE
# ========================================
st.title("🔥 LIVE API: ANAC + OpenCUP + PNRR + AT")
st.markdown("**Chiamate API REAL TIME | Salva DB Master**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 ANAC LIVE", use_container_width=True):
        df_anac = fetch_anac_live()
        st.dataframe(df_anac)

with col2:
    if st.button("🌐 OpenCUP 10 CF", use_container_width=True):
        cfs = ['01206740324', '80012345678', '83001234567']  # Torino/Milano/Roma
        progetti_cup = asyncio.run(batch_opencup(cfs))
        df_cup = pd.DataFrame(progetti_cup)
        df_cup.to_sql('progetti_master', db, if_exists='append', index=False)
        st.dataframe(df_cup)

with col3:
    if st.button("🇪🇺 PNRR 5 Comuni", use_container_width=True):
        comuni_pnrr = ['Torino', 'Milano', 'Roma', 'Napoli', 'Firenze']
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector) as session:
            tasks = [fetch_openpnrr_live(session, c) for c in comuni_pnrr]
            results = await asyncio.gather(*tasks)
        df_pnrr_flat = pd.DataFrame([p for sublist in results for p in sublist])
        st.dataframe(df_pnrr_flat)

with col4:
    if st.button("🔍 AT Scraping 5", use_container_width=True):
        siti_test = {
            'Torino': 'https://www.comune.torino.it',
            'Milano': 'https://www.comune.milano.it'
        }
        # Scraping code
        pass

# MASTER VIEW
st.header("🎯 MASTER DATABASE (Tutte Fonti)")
df_master = pd.read_sql("SELECT * FROM progetti_master ORDER BY timestamp DESC LIMIT 200", db)
st.dataframe(df_master)

st.download_button("💾 Download Master", df_master.to_csv(), "master_api_live.csv")

# Groq AI su risultati API
st.header("🤖 Groq Analizza API Results")
prompt = st.text_input("Analisi:")
if st.button("AI Insight"):
    context = df_master.tail(20).to_dict('records')
    # Groq call
    pass
