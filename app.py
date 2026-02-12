import streamlit as st
import pandas as pd
import requests
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import io
import zipfile
from requests.auth import HTTPBasicAuth

# ==============================
# Streamlit config
# ==============================
st.set_page_config(page_title="API LIVE: 4 Fonti Master (fixed)", layout="wide")

# ==============================
# Helpers
# ==============================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def http_log(source: str, url: str, status: int | None, note: str = ""):
    if "http_log" not in st.session_state:
        st.session_state["http_log"] = []
    st.session_state["http_log"].append(
        {"ts": now_iso(), "source": source, "url": url, "status": status, "note": note}
    )
    # keep last 200
    st.session_state["http_log"] = st.session_state["http_log"][-200:]



def run_async(coro):
    """Run async code safely inside Streamlit."""
    try:
        loop = asyncio.get_running_loop()
        # If we're already in a running loop (rare in Streamlit, but happens in some envs)
        # create a new one.
        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
    except RuntimeError:
        pass
    return asyncio.run(coro)


# ==============================
# DATABASE MASTER
# ==============================
@st.cache_resource
def get_db():
    conn = sqlite3.connect("master_4fonti.db", check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS progetti_master (
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
        )"""
    )
    return conn


db = get_db()


def append_to_db(rows: list[dict]):
    if not rows:
        return
    pd.DataFrame(rows).to_sql("progetti_master", db, if_exists="append", index=False)


# ==============================
# 1) ANAC - CIG anno 2025 (ZIP mensili)
# ==============================

ANAC_CIG_ZIP_BASE = "https://dati.anticorruzione.it/opendata/download/cig/cig_csv_2025_{mm}.zip"


@st.cache_data(ttl=1800)
def fetch_anac_cig_2025(month: str, nrows: int = 5000) -> pd.DataFrame:
    """Scarica ZIP mensile 'CIG anno 2025' e legge il CSV interno."""
    mm = month.zfill(2)
    url = ANAC_CIG_ZIP_BASE.format(mm=mm)

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not csv_names:
        return pd.DataFrame()

    # Prendi il primo CSV
    csv_name = csv_names[0]
    with z.open(csv_name) as f:
        # molti CSV ANAC sono separati da ';' e con encoding variabile.
        # proviamo utf-8, fallback latin1.
        data = f.read()
        for enc in ("utf-8", "latin-1"):
            try:
                s = data.decode(enc)
                buf = io.StringIO(s)
                # sep auto: proviamo ';' poi ','
                try:
                    df_raw = pd.read_csv(buf, sep=";", nrows=nrows, low_memory=False)
                except Exception:
                    buf.seek(0)
                    df_raw = pd.read_csv(buf, sep=",", nrows=nrows, low_memory=False)
                break
            except Exception:
                df_raw = None
        if df_raw is None:
            return pd.DataFrame()

    # Mapping robusto (colonne possono differire nel tempo)
    def col(*names):
        for n in names:
            if n in df_raw.columns:
                return df_raw[n]
        return pd.Series([""] * len(df_raw))

    df = pd.DataFrame()
    df["timestamp"] = now_iso()
    df["fonte"] = "ANAC"
    df["comune"] = col("stazione_appaltante_denominazione", "stazione_appaltante", "sa_denominazione")
    df["identificativo"] = col("cig", "CIG")
    df["titolo"] = col("oggetto", "oggetto_gara", "descrizione")

    imp = col("importo_base_asta", "importo", "importo_gara")
    df["importo"] = pd.to_numeric(imp, errors="coerce").fillna(0.0)

    df["progettista"] = col("aggiudicatario_denominazione", "aggiudicatario", "operatore_economico")
    df["data_evento"] = col("data_pubblicazione", "data", "data_inserimento")
    df["url_dettaglio"] = ""
    df["raw_data"] = ""

    # Salva nel DB
    df.to_sql("progetti_master", db, if_exists="append", index=False)

    # Vista compatta
    out = df[["comune", "identificativo", "titolo", "importo", "progettista"]].copy()
    out.rename(columns={"identificativo": "cig"}, inplace=True)
    return out


# ==============================
# 2) OpenCUP
# 2a) API (richiede credenziali)
# 2b) Fallback OpenData 'Soggetti' (ZIP piccolo)
# ==============================

OPENCUP_API_BASE = "https://api.sogei.it/rgs/opencup/o/extServiceApi/v1/opendataes"
OPENCUP_SOGGETTI_ZIP = "https://www.opencup.gov.it/portale/documents/21195/299152/OpendataSoggetti.zip/411e1e80-bce0-d085-bb96-b8036deb590f"


async def fetch_opencup_api(session: aiohttp.ClientSession, piva_cf: str):
    url = f"{OPENCUP_API_BASE}/soggettotitolare/{piva_cf}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json(content_type=None)
            progetti = []

            # struttura non garantita: proviamo chiavi comuni
            candidates = []
            if isinstance(data, dict):
                for path in [
                    ("data", "progetti"),
                    ("data",),
                    ("progetti",),
                    ("results",),
                ]:
                    cur = data
                    ok = True
                    for k in path:
                        if isinstance(cur, dict) and k in cur:
                            cur = cur[k]
                        else:
                            ok = False
                            break
                    if ok and isinstance(cur, list):
                        candidates = cur
                        break

            for p in candidates[:20]:
                if not isinstance(p, dict):
                    continue
                progetti.append(
                    {
                        "timestamp": now_iso(),
                        "fonte": "OpenCUP(API)",
                        "comune": piva_cf,
                        "identificativo": p.get("codiceCUP") or p.get("cup") or "",
                        "titolo": (p.get("descrizione") or p.get("descrizioneIntervento") or "")[:200],
                        "importo": float(p.get("valoreProgrammato", 0) or 0),
                        "progettista": "",
                        "data_evento": p.get("dataStato") or "",
                        "url_dettaglio": "",
                        "raw_data": json.dumps(p, ensure_ascii=False),
                    }
                )
            return progetti
    except Exception:
        return []


async def batch_opencup_api(cf_list: list[str], auth_user: str, auth_pass: str):
    connector = aiohttp.TCPConnector(limit=10)
    auth = aiohttp.BasicAuth(auth_user, auth_pass)
    async with aiohttp.ClientSession(connector=connector, auth=auth) as session:
        tasks = [fetch_opencup_api(session, cf) for cf in cf_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    flat = []
    for r in results:
        if isinstance(r, list):
            flat.extend(r)
    return flat


@st.cache_data(ttl=86400)
def load_opencup_soggetti_zip() -> pd.DataFrame:
    """Carica OpenData 'Soggetti' (ZIP piccolo) e restituisce un dataframe."""
    r = requests.get(OPENCUP_SOGGETTI_ZIP, timeout=60)
    http_log('OpenCUP(OpenData)', OPENCUP_SOGGETTI_ZIP, r.status_code)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not csv_names:
        return pd.DataFrame()

    with z.open(csv_names[0]) as f:
        raw = f.read()

    # encoding dichiarato spesso ASCII; separatore '|' (Pipe)
    text = raw.decode("ascii", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep="|", low_memory=False)
    return df


def fetch_opencup_soggetti(cf_list: list[str], max_rows_per_cf: int = 50) -> list[dict]:
    """Fallback: usa dataset 'Soggetti' per costruire record 'progetti_master' (anagrafiche)."""
    df = load_opencup_soggetti_zip()
    if df.empty:
        return []

    # euristica: trova la colonna CF/PIVA
    cf_col = None
    for cand in ["codice_fiscale", "cf", "piva_cf", "partita_iva", "piva", "codiceFiscale"]:
        if cand in df.columns:
            cf_col = cand
            break

    # se non trovata, prova a cercare per nome colonna simile
    if cf_col is None:
        for c in df.columns:
            if "fisc" in c.lower() or "iva" in c.lower():
                cf_col = c
                break

    if cf_col is None:
        return []

    out = []
    for cf in cf_list:
        sub = df[df[cf_col].astype(str).str.strip() == str(cf).strip()].head(max_rows_per_cf)
        if sub.empty:
            continue

        for _, row in sub.iterrows():
            comune = row.get("denominazione", "")
            if not comune and "ragione" in " ".join(df.columns).lower():
                # lascia vuoto
                pass
            out.append(
                {
                    "timestamp": now_iso(),
                    "fonte": "OpenCUP(OpenData:Soggetti)",
                    "comune": comune if isinstance(comune, str) else str(comune),
                    "identificativo": str(cf),
                    "titolo": "Anagrafica soggetto titolare/richiedente",
                    "importo": 0.0,
                    "progettista": "",
                    "data_evento": "",
                    "url_dettaglio": "https://www.opencup.gov.it/",
                    "raw_data": json.dumps(row.to_dict(), ensure_ascii=False),
                }
            )

    return out


# ==============================
# 3) OpenPNRR API (parametro 'territori')
# ==============================

OPENPNRR_API = "https://openpnrr.it/api/v1/progetti"


def fetch_openpnrr(comune: str, page_size: int = 20, validato: bool = False) -> list[dict]:
    params = {
        "territori": comune,
        "page_size": page_size,
    }
    if validato:
        # Nel contratto OpenAPI è presente il filtro `validato`.
        # Alcune installazioni accettano "true/false" come stringa.
        params["validato"] = "true"

    auth = None
    # Se l'istanza richiede BasicAuth, puoi impostare OPENPNRR_USER / OPENPNRR_PASS in secrets.
    if "OPENPNRR_USER" in st.secrets and "OPENPNRR_PASS" in st.secrets:
        auth = HTTPBasicAuth(st.secrets["OPENPNRR_USER"], st.secrets["OPENPNRR_PASS"])

    r = requests.get(OPENPNRR_API, params=params, timeout=30, auth=auth)
    http_log('OpenPNRR', r.url, r.status_code)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

    data = r.json()
    results = data.get("results", []) if isinstance(data, dict) else []

    out = []
    for p in results[:page_size]:
        if not isinstance(p, dict):
            continue
        out.append(
            {
                "timestamp": now_iso(),
                "fonte": "OpenPNRR",
                "comune": comune,
                "identificativo": str(p.get("id") or p.get("id_progetto") or ""),
                "titolo": (p.get("descrizione") or p.get("titolo") or p.get("descrizione_progetto") or "")[:200],
                "importo": float(p.get("importo", 0) or 0),
                "progettista": "",
                "data_evento": p.get("data") or p.get("data_avvio") or "",
                "url_dettaglio": p.get("url") or p.get("link") or "",
                "raw_data": json.dumps(p, ensure_ascii=False),
            }
        )
    return out


# ==============================
# 4) AT Scraping
# ==============================

async def scrape_at_live(session: aiohttp.ClientSession, sito: str, comune: str):
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
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                if resp.status != 200:
                    http_log('AT', url, resp.status)
                    continue
                http_log('AT', url, resp.status)
                html = await resp.text(errors="ignore")
                soup = BeautifulSoup(html, "html.parser")

                link_elems = soup.find_all("a", href=keywords_href)
                text_hits = soup.find_all(string=keywords_text)

                if not link_elems and not text_hits:
                    continue

                titoli = [t.strip() for t in text_hits if isinstance(t, str) and len(t.strip()) > 10]
                titoli = titoli[:10]

                risultati.append(
                    {
                        "timestamp": now_iso(),
                        "fonte": "AT",
                        "comune": comune,
                        "identificativo": "",
                        "titolo": f"Bandi/AT trovati su {url}",
                        "importo": 0.0,
                        "progettista": "",
                        "data_evento": "",
                        "url_dettaglio": url,
                        "raw_data": json.dumps(
                            {
                                "count_link": len(link_elems),
                                "sample_links": [
                                    {
                                        "text": (a.get_text(strip=True) or "")[:120],
                                        "href": urljoin(url, a.get("href", "")) if a.get("href") else "",
                                    }
                                    for a in link_elems[:10]
                                ],
                                "titoli": titoli,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
        except Exception:
            continue

    if not risultati:
        risultati.append(
            {
                "timestamp": now_iso(),
                "fonte": "AT",
                "comune": comune,
                "identificativo": "",
                "titolo": "Nessun bando/AT rilevato",
                "importo": 0.0,
                "progettista": "",
                "data_evento": "",
                "url_dettaglio": sito,
                "raw_data": "",
            }
        )

    return risultati


async def batch_at(siti_dict: dict):
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [scrape_at_live(session, sito, comune) for comune, sito in siti_dict.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    flat = []
    for r in results:
        if isinstance(r, list):
            flat.extend(r)
    return flat


# ==============================
# UI
# ==============================

st.title("🔥 LIVE API: ANAC + OpenCUP + OpenPNRR + AT (fixed)")
st.caption("Chiamate REAL TIME + salvataggio su SQLite (tabella: progetti_master)")

with st.sidebar:
    st.header("⚙️ Impostazioni")

    st.subheader("ANAC")
    anac_month = st.selectbox("Mese 2025 (ZIP CIG)", [f"{i:02d}" for i in range(1, 13)], index=0)
    anac_nrows = st.slider("Righe da leggere (demo)", min_value=200, max_value=20000, value=2000, step=200)

    st.subheader("OpenCUP")
    opencup_mode = st.radio(
        "Modalità", ["OpenData (Soggetti.zip)", "API (richiede credenziali)"], index=0
    )

    st.subheader("OpenPNRR")
    pnrr_page_size = st.slider("Page size", min_value=5, max_value=50, value=15, step=5)
    pnrr_custom = st.text_input("Aggiungi comuni (separati da virgola)", value="")
    pnrr_use_validation = st.checkbox("Solo progetti validati (se supportato)", value=False)

    st.subheader("AT (scraping)")
    use_custom_sites = st.checkbox("Usa elenco siti personalizzato", value=True)

    st.subheader("Debug")
    show_raw = st.checkbox("Mostra raw_data", value=False)
    show_http_debug = st.checkbox("Mostra log HTTP", value=False)
col1, col2, col3, col4 = st.columns(4)

# ---- ANAC ----
with col1:
    if st.button("📊 ANAC (CIG 2025)", use_container_width=True):
        with st.spinner("Scarico ZIP ANAC e leggo CSV..."):
            try:
                df_anac = fetch_anac_cig_2025(anac_month, nrows=anac_nrows)
                if df_anac.empty:
                    st.warning("Nessun dato letto dal CSV nel ZIP.")
                else:
                    st.dataframe(df_anac, use_container_width=True)
            except Exception as e:
                st.error(f"ANAC errore: {e}")

# ---- OpenCUP ----
with col2:
    st.write("")
    cfs_txt = st.text_area(
        "CF/P.IVA soggetto titolare (uno per riga)",
        value="01206740324\n80012345678\n83001234567",
        height=120,
        label_visibility="visible",
    )

    if st.button("🌐 OpenCUP", use_container_width=True):
        cf_list = [x.strip() for x in cfs_txt.splitlines() if x.strip()]
        if not cf_list:
            st.warning("Inserisci almeno un CF/P.IVA.")
        else:
            if opencup_mode.startswith("API"):
                if "OPENCUP_USER" not in st.secrets or "OPENCUP_PASS" not in st.secrets:
                    st.error(
                        "Mancano credenziali OpenCUP API. Imposta in `.streamlit/secrets.toml`:\n"
                        "OPENCUP_USER=...\nOPENCUP_PASS=...\n\n"
                        "Oppure usa la modalità OpenData (Soggetti.zip)."
                    )
                else:
                    with st.spinner("Chiamo OpenCUP API..."):
                        rows = run_async(
                            batch_opencup_api(
                                cf_list,
                                st.secrets["OPENCUP_USER"],
                                st.secrets["OPENCUP_PASS"],
                            )
                        )
                        if not rows:
                            st.warning("Nessun risultato (API).")
                        else:
                            append_to_db(rows)
                            df = pd.DataFrame(rows)
                            if not show_raw and "raw_data" in df.columns:
                                df = df.drop(columns=["raw_data"])
                            st.dataframe(df, use_container_width=True)
            else:
                with st.spinner("Scarico OpenData Soggetti.zip e filtro..."):
                    try:
                        rows = fetch_opencup_soggetti(cf_list)
                        if not rows:
                            st.warning("Nessuna anagrafica trovata per quei CF/PIVA.")
                        else:
                            append_to_db(rows)
                            df = pd.DataFrame(rows)
                            if not show_raw and "raw_data" in df.columns:
                                df = df.drop(columns=["raw_data"])
                            st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"OpenCUP OpenData errore: {e}")

# ---- OpenPNRR ----
with col3:
    comuni_pnrr = st.multiselect(
        "Comuni (territori)",
        ["Torino", "Milano", "Roma", "Napoli", "Firenze", "Bologna", "Genova"],
        default=["Torino", "Milano", "Roma"],
    )

    # Aggiunta rapida da sidebar (virgole)
    if pnrr_custom.strip():
        extra = [x.strip() for x in pnrr_custom.split(",") if x.strip()]
        for x in extra:
            if x not in comuni_pnrr:
                comuni_pnrr.append(x)

    if st.button("🇪🇺 OpenPNRR", use_container_width=True):
        if not comuni_pnrr:
            st.warning("Seleziona almeno un comune.")
        else:
            all_rows = []
            with st.spinner("Chiamo OpenPNRR API..."):
                for c in comuni_pnrr:
                    try:
                        all_rows.extend(fetch_openpnrr(c, page_size=pnrr_page_size, validato=pnrr_use_validation))
                    except Exception as e:
                        st.error(f"OpenPNRR ({c}) errore: {e}")

            if not all_rows:
                st.warning("Nessun progetto recuperato.")
            else:
                append_to_db(all_rows)
                df = pd.DataFrame(all_rows)
                if not show_raw and "raw_data" in df.columns:
                    df = df.drop(columns=["raw_data"])
                st.dataframe(df, use_container_width=True)

# ---- AT Scraping ----
with col4:
    default_sites = {
        "Torino": "https://www.comune.torino.it",
        "Milano": "https://www.comune.milano.it",
        "Roma": "https://www.comune.roma.it",
        "Napoli": "https://www.comune.napoli.it",
        "Firenze": "https://www.comune.fi.it",
    }

    if "at_sites" not in st.session_state:
        st.session_state["at_sites"] = pd.DataFrame(
            [{"comune": k, "sito": v} for k, v in default_sites.items()]
        )

    if use_custom_sites:
        st.write("Elenco siti (modificabile)")
        edited = st.data_editor(
            st.session_state["at_sites"],
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "comune": st.column_config.TextColumn("Comune"),
                "sito": st.column_config.TextColumn("Sito (https://...)"),
            },
            hide_index=True,
        )
        # Persist
        st.session_state["at_sites"] = edited

        siti_dict = {
            str(r.get("comune", "")).strip(): str(r.get("sito", "")).strip()
            for r in edited.to_dict("records")
            if str(r.get("comune", "")).strip() and str(r.get("sito", "")).strip()
        }
    else:
        st.caption("Usando elenco siti predefinito.")
        siti_dict = default_sites

    if st.button("🔍 AT Scraping", use_container_width=True):
        if not siti_dict:
            st.warning("Aggiungi almeno un sito valido.")
        else:
            with st.spinner("Scansiono pagine Trasparenza/Albo..."):
                rows = run_async(batch_at(siti_dict))
                append_to_db(rows)
                df = pd.DataFrame(rows)
                if not show_raw and "raw_data" in df.columns:
                    df = df.drop(columns=["raw_data"])
                st.dataframe(df, use_container_width=True)

if show_http_debug and "http_log" in st.session_state and st.session_state["http_log"]:
    with st.expander("🧾 Log HTTP (ultime chiamate)", expanded=False):
        st.dataframe(pd.DataFrame(st.session_state["http_log"][::-1]), use_container_width=True)

st.divider()

# MASTER VIEW
st.header("🎯 MASTER DATABASE (tutte le fonti)")

try:
    df_master = pd.read_sql(
        "SELECT * FROM progetti_master ORDER BY datetime(timestamp) DESC LIMIT 200",
        db,
    )
    if not show_raw and "raw_data" in df_master.columns:
        df_view = df_master.drop(columns=["raw_data"])
    else:
        df_view = df_master

    st.dataframe(df_view, use_container_width=True)
    st.download_button(
        "💾 Download Master (CSV)",
        df_master.to_csv(index=False).encode("utf-8"),
        "master_api_live.csv",
        mime="text/csv",
    )
except Exception as e:
    st.warning(f"Nessun dato nella tabella progetti_master ancora. Dettaglio: {e}")

st.divider()

# Optional: Groq AI
st.header("🤖 Groq (opzionale) – Analizza risultati")

prompt = st.text_input("Analisi (es: 'trova comuni con importi più alti'):")

if st.button("AI Insight"):
    try:
        df_master = pd.read_sql(
            "SELECT * FROM progetti_master ORDER BY datetime(timestamp) DESC LIMIT 200",
            db,
        )
        if df_master.empty:
            st.warning("Nessun dato in master per l'analisi AI.")
        else:
            if "GROQ_API_KEY" not in st.secrets:
                st.error("Imposta GROQ_API_KEY in .streamlit/secrets.toml")
            else:
                from groq import Groq

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
