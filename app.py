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
    """ANAC dataset appalti 2025 LIVE (demo, primo CSV trovato)."""
    try:
        url = "https://dati.anticorruzione.it/opendata/dataset/ocds-appalti-ordinari-2025"
        resp = requests.get(url, timeout=30)

        soup = BeautifulSoup(resp.text, 'html.parser')
        csv_links = soup.find_all('a', href=re.compile(r'\.csv$', re.I))

        if not csv_links:
            st.warning("Nessun CSV trovato nella pagina ANAC (controlla HTML reale).")
            return pd.DataFrame()

        csv_url = csv_links[0]['href']
        if not csv_url.startswith("http"):
            # se è relativo, lo agganciamo alla base
            csv_url = requests.compat.urljoin(url, csv_url)

        # ATTENZIONE: nrows per non scaricare tutto in demo
        df_raw = pd.read_csv(csv_url, nrows=200)

        # Qui andrebbe mappato sui campi veri del dataset ANAC
        # Per ora estraiamo in modo prudente
        df = pd.DataFrame()
        df["comune"] = df_raw.get("stazione_appaltante_denominazione", pd.Series([""] * len(df_raw)))
        df["cig"] = df_raw.get("cig", pd.Series([""] * len(df_raw)))
        df["titolo"] = df_raw.get("oggetto", pd.Series([""] * len(df_raw)))
        df["importo"] = df_raw.get("importo_base_asta", pd.Series([0] * len(df_raw)))
        df["progettista"] = df_raw.get("aggiudicatario_denominazione", pd.Series([""] * len(df_raw)))

        df["fonte"] = "ANAC"
        df["timestamp"] = datetime.now().isoformat()
        df["data_evento"] = df_raw.get("data_pubblicazione", pd.Series([""] * len(df_raw)))
        df["identificativo"] = df["cig"]
        df["url_dettaglio"] = ""  # nel CSV non sempre c'è un link diretto
        df["raw_data"] = ""       # potresti salvare json.dumps(riga) se vuoi

        df.to_sql("progetti_master", db, if_exists="append", index=False)
        return df[["comune", "cig", "titolo", "importo", "progettista"]]
    except Exception as e:
        st.error(f"ANAC: {e}")
        return pd.DataFrame()

# ========================================
# 2. OPENCUP API LIVE (CF Comuni)
# ========================================
async def fetch_opencup_live(session, cf_comune: str):
    """OpenCUP API reale per un CF di soggetto titolare."""
    url = f"https://api.sogei.it/rgs/opencup/o/extServiceApi/v1/opendataes/soggettotitolare/{cf_comune}"
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                progetti = []
                # struttura basata sulla doc generica, verifica chiavi reali
                for p in data.get("data", {}).get("progetti", [])[:5]:
                    progetti.append({
                        "timestamp": datetime.now().isoformat(),
                        "fonte": "OpenCUP",
                        "comune": cf_comune,  # se vuoi mappare a nome comune, serve mapping CF->nome
                        "identificativo": p.get("codiceCUP"),
                        "titolo": (p.get("descrizione") or "")[:150],
                        "importo": float(p.get("valoreProgrammato", 0) or 0),
                        "progettista": "",  # spesso non c'è nel CUP
                        "data_evento": p.get("dataStato") or "",
                        "url_dettaglio": "",  # es: link portale OpenCUP se presente
                        "raw_data": json.dumps(p, ensure_ascii=False)
                    })
                return progetti
    except Exception as e:
        st.error(f"OpenCUP ({cf_comune}): {e}")
    return []

async def batch_opencup(cf_list):
    """Batch OpenCUP 10 comuni (CF)."""
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_opencup_live(session, cf) for cf in cf_list[:10]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    flat = []
    for r in results:
        if isinstance(r, list):
            flat.extend(r)
    return flat

# ========================================
# 3. OPENPNRR API LIVE
# ========================================
async def fetch_openpnrr_live(session, comune: str):
    """OpenPNRR API reale per un comune."""
    url = f"https://openpnrr.it/api/v1/progetti/?comune={comune.replace(' ', '%20')}&limit=10"
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status == 200:
                data = await resp.json()
                progetti = []
                for p in data.get("results", [])[:5]:
                    progetti.append({
                        "timestamp": datetime.now().isoformat(),
                        "fonte": "OpenPNRR",
                        "comune": comune,
                        "identificativo": p.get("id_progetto") or p.get("id"),
                        "titolo": (p.get("descrizione_progetto") or p.get("descrizione") or "")[:150],
                        "importo": float(p.get("importo", 0) or 0),
                        "progettista": "",  # in PNRR di solito non c'è "progettista"
                        "data_evento": p.get("data_avvio") or p.get("data_approvazione") or "",
                        "url_dettaglio": p.get("link") or "",
                        "raw_data": json.dumps(p, ensure_ascii=False)
                    })
                return progetti
    except Exception as e:
        st.error(f"OpenPNRR ({comune}): {e}")
    return []

async def batch_openpnrr(comuni_list):
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_openpnrr_live(session, c) for c in comuni_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    flat = []
    for r in results:
        if isinstance(r, list):
            flat.extend(r)
    return flat

# ========================================
# 4. AT SCRAPING (implementazione)
# ========================================
async def scrape_at_live(session, sito: str, comune: str):
    """
    Scraping Amministrazione Trasparente / Bandi / Albo Pretorio
    Restituisce un record compatibile con 'progetti_master'
    (fonte='AT', raw_data=titoli/links trovati).
    """
    paths = [
        "/amministrazione-trasparente/",
        "/trasparenza/",
        "/amministrazione-trasparente/bandi-di-gara-e-contratti/",
        "/bandi-di-gara/",
        "/gare-appalti/",
        "/albo-pretorio/",
        "/bandi",
    ]
    keywords_href = re.compile(r"(bandi?|gare|appalt|contratt|albo)", re.I)
    keywords_text = re.compile(r"(bando|gara|appalto|aggiudicat|affidament)", re.I)

    risultati = []

    for p in paths:
        url = urljoin(sito.rstrip("/") + "/", p.lstrip("/"))
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    continue
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")

                link_elems = soup.find_all("a", href=keywords_href)
                text_hits = soup.find_all(string=keywords_text)

                if not link_elems and not text_hits:
                    continue

                titoli = [t.strip() for t in text_hits if isinstance(t, str) and len(t.strip()) > 10]
                titoli = titoli[:10]

                risultati.append({
                    "timestamp": datetime.now().isoformat(),
                    "fonte": "AT",
                    "comune": comune,
                    "identificativo": "",  # niente CIG/CUP qui
                    "titolo": f"Bandi/AT trovati su {url}",
                    "importo": 0.0,
                    "progettista": "",
                    "data_evento": "",
                    "url_dettaglio": url,
                    "raw_data": json.dumps({
                        "count_link": len(link_elems),
                        "titoli": titoli
                    }, ensure_ascii=False)
                })
        except Exception as e:
            # non bloccare tutto per un comune
            continue

    # Se niente trovato, restituisci un record "vuoto" per tracciare il tentativo
    if not risultati:
        risultati.append({
            "timestamp": datetime.now().isoformat(),
            "fonte": "AT",
            "comune": comune,
            "identificativo": "",
            "titolo": "Nessun bando/AT rilevato",
            "importo": 0.0,
            "progettista": "",
            "data_evento": "",
            "url_dettaglio": sito,
            "raw_data": ""
        })

    return risultati

async def batch_at(siti_dict: dict):
    """Scraping AT per più comuni in parallelo."""
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [scrape_at_live(session, sito, comune) for comune, sito in siti_dict.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    flat = []
    for r in results:
        if isinstance(r, list):
            flat.extend(r)
    return flat

# ========================================
# DASHBOARD 4 API LIVE
# ========================================
st.title("🔥 LIVE API: ANAC + OpenCUP + PNRR + AT")
st.markdown("**Chiamate API REAL TIME | Salva DB Master (tabella: progetti_master)**")

col1, col2, col3, col4 = st.columns(4)

# ---- ANAC ----
with col1:
    if st.button("📊 ANAC LIVE", use_container_width=True):
        df_anac = fetch_anac_live()
        if df_anac.empty:
            st.warning("Nessun dato ANAC caricato.")
        else:
            st.dataframe(df_anac)

# ---- OpenCUP ----
with col2:
    if st.button("🌐 OpenCUP 10 CF", use_container_width=True):
        cfs = ['01206740324', '80012345678', '83001234567']  # Esempi Torino/Milano/Roma
        progetti_cup = asyncio.run(batch_opencup(cfs))
        if not progetti_cup:
            st.warning("Nessun progetto CUP recuperato.")
        else:
            df_cup = pd.DataFrame(progetti_cup)
            df_cup.to_sql('progetti_master', db, if_exists='append', index=False)
            st.dataframe(df_cup)

# ---- OpenPNRR ----
with col3:
    if st.button("🇪🇺 PNRR 5 Comuni", use_container_width=True):
        comuni_pnrr = ['Torino', 'Milano', 'Roma', 'Napoli', 'Firenze']
        progetti_pnrr = asyncio.run(batch_openpnrr(comuni_pnrr))
        if not progetti_pnrr:
            st.warning("Nessun progetto PNRR recuperato.")
        else:
            df_pnrr_flat = pd.DataFrame(progetti_pnrr)
            df_pnrr_flat.to_sql('progetti_master', db, if_exists='append', index=False)
            st.dataframe(df_pnrr_flat)

# ---- AT Scraping ----
with col4:
    if st.button("🔍 AT Scraping 5", use_container_width=True):
        siti_test = {
            'Torino': 'https://www.comune.torino.it',
            'Milano': 'https://www.comune.milano.it',
            'Roma': 'https://www.comune.roma.it',
            'Napoli': 'https://www.comune.napoli.it',
            'Firenze': 'https://www.comune.fi.it',
        }
        risultati_at = asyncio.run(batch_at(siti_test))
        df_at = pd.DataFrame(risultati_at)
        df_at.to_sql('progetti_master', db, if_exists='append', index=False)
        st.dataframe(df_at)

# MASTER VIEW
st.header("🎯 MASTER DATABASE (Tutte Fonti)")
try:
    df_master = pd.read_sql(
        "SELECT * FROM progetti_master ORDER BY datetime(timestamp) DESC LIMIT 200",
        db
    )
    st.dataframe(df_master)
    st.download_button("💾 Download Master", df_master.to_csv(index=False), "master_api_live.csv")
except Exception as e:
    st.warning(f"Nessun dato nella tabella progetti_master ancora. Dettaglio: {e}")

# Groq AI su risultati API
st.header("🤖 Groq Analizza API Results")
prompt = st.text_input("Analisi (es: 'trova comuni con importi più alti'):")

if st.button("AI Insight"):
    try:
        df_master = pd.read_sql(
            "SELECT * FROM progetti_master ORDER BY datetime(timestamp) DESC LIMIT 200",
            db
        )
        if df_master.empty:
            st.warning("Nessun dato in master per l'analisi AI.")
        else:
            context = df_master.tail(50).to_dict("records")

            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            full_prompt = f"""
Sei un esperto di appalti pubblici italiani, PNRR e trasparenza.
Analizza i seguenti record (ANAC, OpenCUP, OpenPNRR, AT scraping):

{json.dumps(context, ensure_ascii=False)[:8000]}

Domanda dell'utente:
{prompt}

Rispondi in italiano, in modo sintetico ma tecnico, con:
- Riepilogo numerico (quanti comuni, quante fonti)
- Top 5 comuni per importo
- Eventuali anomalie
- Suggerimenti di indagine OSINT
"""

            resp = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.2,
                max_tokens=1200,
            )
            st.markdown("### 🧠 Insight Groq")
            st.write(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"Errore Groq o DB: {e}")
