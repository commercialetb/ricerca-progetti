"""
Script aggiornato per recuperare dati da ANAC, OpenCUP, OpenPNRR e sezione
"Amministrazione Trasparente".  La versione originale cercava di scaricare
dataset non più disponibili o richiedeva credenziali che non erano
specificate.  Questo modulo mostra come interrogare le fonti open data
aggiornate al 2026.

Principali correzioni:

* **ANAC** – il dataset "ocds-appalti-ordinari-2025" non contiene file CSV
  direttamente linkati nel portale.  Al suo posto si utilizza il dataset
  "CIG anno 2025" che offre file mensili compressi in ZIP.  Il primo file
  (cig_csv_2025_01.zip) viene scaricato, decompresso in memoria e
  convertito in un DataFrame.  Per evitare di scaricare l'intero
  database (decine di MB), si limitano le righe lette utilizzando il
  parametro ``nrows`` di pandas.

* **OpenCUP** – l'API REST documentata sul sito richiede
  autenticazione (restituisce ``401 Unauthorized`` se chiamata senza
  credenziali).  In assenza di un token/chiave, è possibile usare i
  dataset open data rilasciati in formato CSV/ZIP sul sito del
  Dipartimento per la programmazione economica.  Qui si fornisce una
  funzione che scarica l'archivio "Soggetti" (contiene le anagrafiche
  dei soggetti titolari) e filtra per i codici fiscali forniti; se si
  dispone di credenziali per l'API REST, è possibile reinserire il
  client aiohttp e passare ``auth=aiohttp.BasicAuth(...)``.

* **OpenPNRR** – l'end‑point usato in origine (``/api/v1/progetti/?comune=``)
  non esiste.  La piattaforma offre un'esplorazione API documentata
  all'indirizzo ``/api/v1/schema/``, con parametri quali ``territori``,
  ``organizzazioni`` e ``descrizione``.  Per semplicità, si utilizza
  l'API REST senza autenticazione richiamando ``/api/v1/progetti`` e
  filtrando per territorio (comune) quando possibile.  In alternativa,
  è possibile scaricare i file CSV elencati nella pagina OpenPNRR
  ``/opendata/``, ma questi file potrebbero essere ospitati su S3 e
  richiedere accesso.  La funzione ``fetch_openpnrr_live`` tenta una
  chiamata all'API e ritorna un elenco sintetico dei progetti.

* **Amministrazione Trasparente** – la logica di scraping è stata
  mantenuta ma migliorata per gestire meglio eventuali errori.  Viene
  restituito sempre un record di log, anche quando non si trovano
  bandi.

Nota: le chiamate di rete possono richiedere tempi e larghezza di
 banda elevati.  Nel codice seguente sono impostati limiti conservativi
 per evitare timeout o download troppo pesanti.  È sempre consigliato
 salvare i risultati su un database locale per riutilizzarli in altre
 analisi.
"""

import asyncio
import json
import re
import sqlite3
from datetime import datetime
from io import BytesIO
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------
def get_db(path: str = "master_4fonti.db") -> sqlite3.Connection:
    """Crea (se necessario) e restituisce una connessione SQLite thread‑safe.

    Args:
        path: percorso del file database.

    Returns:
        sqlite3.Connection: connessione aperta al database.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS progetti_master (
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
        )
        """
    )
    return conn


# ---------------------------------------------------------------------------
# 1. ANAC – dataset CIG anno 2025 (CSV ZIP)
# ---------------------------------------------------------------------------
def fetch_anac_cig_2025(nrows: int = 500) -> pd.DataFrame:
    """Scarica una porzione del dataset ANAC "CIG anno 2025".

    Nel portale open data di ANAC il dataset relativo agli appalti del
    2025 viene distribuito in 12 file ZIP (uno per mese).  Questa
    funzione scarica il primo file (gennaio) e restituisce alcune
    colonne rilevanti.  Se necessario si può estendere al download di
    più mesi.

    Args:
        nrows: numero massimo di righe da leggere dal CSV contenuto nel ZIP.

    Returns:
        DataFrame con colonne mappate su comune, CIG, titolo, importo e
        aggiudicatario.
    """
    base_url = (
        "https://dati.anticorruzione.it/opendata/download/dataset/cig-2025"
        "/filesystem/cig_csv_2025_01.zip"
    )
    try:
        resp = requests.get(base_url, timeout=60)
        resp.raise_for_status()
        zip_bytes = BytesIO(resp.content)
        # Estrai il primo file CSV all'interno del ZIP
        import zipfile

        with zipfile.ZipFile(zip_bytes) as zf:
            # prendi il primo file csv (di solito unico)
            csv_name = next(
                (name for name in zf.namelist() if name.lower().endswith(".csv")),
                None,
            )
            if not csv_name:
                return pd.DataFrame()
            with zf.open(csv_name) as f:
                df_raw = pd.read_csv(f, sep=";", nrows=nrows, encoding="latin1")
    except Exception as e:
        # se il download fallisce restituisce DataFrame vuoto
        print(f"Errore ANAC CIG 2025: {e}")
        return pd.DataFrame()
    # Mappatura campi; i nomi delle colonne possono cambiare nel tempo.
    df = pd.DataFrame()
    df["comune"] = df_raw.get(
        "stazione_appaltante_denominazione",
        df_raw.get("denominazione_stazione_appaltante", pd.Series([""] * len(df_raw))),
    )
    df["cig"] = df_raw.get("cig", pd.Series([""] * len(df_raw)))
    df["titolo"] = df_raw.get(
        "oggetto",
        df_raw.get("oggetto_principale", pd.Series([""] * len(df_raw))),
    )
    importo_col = (
        df_raw.get("importo_base_asta")
        or df_raw.get("importo_appalto")
        or df_raw.get("importo_a_base_di_gara")
    )
    df["importo"] = importo_col.fillna(0).astype(float)
    df["progettista"] = df_raw.get(
        "aggiudicatario_denominazione",
        df_raw.get("denominazione_aggiudicatario", pd.Series([""] * len(df_raw))),
    )
    df["fonte"] = "ANAC"
    df["timestamp"] = datetime.now().isoformat()
    df["data_evento"] = df_raw.get(
        "data_pubblicazione",
        df_raw.get("data_inizio_procedura", pd.Series([""] * len(df_raw))),
    )
    df["identificativo"] = df["cig"]
    df["url_dettaglio"] = ""
    df["raw_data"] = ""
    return df[["comune", "cig", "titolo", "importo", "progettista"]]


# ---------------------------------------------------------------------------
# 2. OpenCUP – dataset soggetti titolari (CSV ZIP)
# ---------------------------------------------------------------------------
async def fetch_opencup_from_csv(session: aiohttp.ClientSession, cf_comune: str, nrows: int = 100) -> List[dict]:
    """Scarica l'anagrafica dei soggetti titolari da open data e filtra per CF.

    Il sito OpenCUP pubblica dataset compressi con i dati anagrafici dei
    soggetti titolari/beneficiari.  In mancanza di credenziali per l'API
    REST (che restituisce 401 se non autenticati【513578321398614†L0-L2】), questa funzione
    scarica l'archivio "Soggetti" e restituisce i record del codice
    fiscale richiesto.

    Args:
        session: istanza aiohttp.
        cf_comune: codice fiscale del comune/soggetto titolare.
        nrows: numero massimo di righe da leggere dal CSV (per limitare il
            trasferimento dati).

    Returns:
        Elenco di dizionari compatibile con la tabella ``progetti_master``.
    """
    url_zip = "https://www.opencup.gov.it/portale/opencup-data/soggetti.zip"
    try:
        async with session.get(url_zip, timeout=60) as resp:
            if resp.status != 200:
                return []
            content = await resp.read()
        import zipfile
        import csv
        from io import TextIOWrapper
        zf = zipfile.ZipFile(BytesIO(content))
        csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if not csv_name:
            return []
        with zf.open(csv_name) as f:
            reader = csv.DictReader(TextIOWrapper(f, encoding="latin1"), delimiter="|")
            progetti = []
            count = 0
            for row in reader:
                if cf_comune and row.get("codice_fiscale") == cf_comune:
                    progetti.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "fonte": "OpenCUP",
                            "comune": row.get("denominazione", ""),
                            "identificativo": row.get("codice_fiscale", ""),
                            "titolo": row.get("tipologia_soggetto", ""),
                            "importo": 0.0,
                            "progettista": row.get("denominazione", ""),
                            "data_evento": "",
                            "url_dettaglio": "https://opencup.gov.it/portale/progetto/-/cup/{}".format(row.get("codice_fiscale")),
                            "raw_data": json.dumps(row, ensure_ascii=False),
                        }
                    )
                    count += 1
                if count >= nrows:
                    break
        return progetti
    except Exception:
        return []


async def batch_opencup(cf_list: Iterable[str]) -> List[dict]:
    """Esegue in parallelo il download dei dati OpenCUP per più codici fiscali."""
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_opencup_from_csv(session, cf) for cf in cf_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    flat: List[dict] = []
    for res in results:
        if isinstance(res, list):
            flat.extend(res)
    return flat


# ---------------------------------------------------------------------------
# 3. OpenPNRR – utilizzo dell'API REST ufficiale
# ---------------------------------------------------------------------------
async def fetch_openpnrr_live(session: aiohttp.ClientSession, comune: str, max_results: int = 5) -> List[dict]:
    """Interroga l'API OpenPNRR e restituisce i progetti per un comune.

    L'API ufficiale di OpenPNRR consente di filtrare i progetti per
    diversi parametri.  Non esiste un parametro ``comune`` ma è
    possibile usare ``territori`` passando il nome del territorio o il
    suo codice ISTAT.  Se la richiesta restituisce troppi dati, si
    utilizza il parametro ``page_size`` per limitarne la quantità.

    Args:
        session: ClientSession aiohttp condiviso.
        comune: nome del comune su cui filtrare (es. "Torino").
        max_results: numero massimo di record da restituire.

    Returns:
        Lista di dizionari con informazioni principali sui progetti.
    """
    # Prepara query: utilizziamo la chiave ``territori`` con il nome del comune
    # e limitiamo la pagina a 1 con page_size adeguato.  Se l'endpoint
    # richiedesse autenticazione Basic, si dovrebbe passare l'argomento
    # auth=aiohttp.BasicAuth(user, password).
    params = {
        "territori": comune,
        "page": 1,
        "page_size": max_results,
    }
    url = "https://openpnrr.it/api/v1/progetti"
    try:
        async with session.get(url, params=params, timeout=30) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            results = data.get("results", [])
    except Exception:
        return []
    progetti: List[dict] = []
    for p in results:
        progetti.append(
            {
                "timestamp": datetime.now().isoformat(),
                "fonte": "OpenPNRR",
                "comune": comune,
                "identificativo": p.get("id_progetto") or p.get("id"),
                "titolo": (p.get("descrizione_progetto") or p.get("descrizione") or "")[:150],
                "importo": float(p.get("importo") or p.get("importo_totale") or 0),
                "progettista": "",  # non disponibile nell'API
                "data_evento": p.get("data_avvio") or p.get("data_approvazione") or "",
                "url_dettaglio": p.get("link") or "",
                "raw_data": json.dumps(p, ensure_ascii=False),
            }
        )
    return progetti


async def batch_openpnrr(comuni_list: Iterable[str], max_results_per_comune: int = 5) -> List[dict]:
    """Esegue più chiamate OpenPNRR in parallelo per una lista di comuni."""
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_openpnrr_live(session, comune, max_results=max_results_per_comune)
            for comune in comuni_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    flat: List[dict] = []
    for res in results:
        if isinstance(res, list):
            flat.extend(res)
    return flat


# ---------------------------------------------------------------------------
# 4. Amministrazione Trasparente – scraping
# ---------------------------------------------------------------------------
async def scrape_at_live(session: aiohttp.ClientSession, sito: str, comune: str) -> List[dict]:
    """Scraping generico della sezione Amministrazione Trasparente di un comune.

    La funzione tenta diversi percorsi standard per trovare pagine con bandi,
    gare o contratti.  Se trova titoli o link coerenti, restituisce
    informazioni di sintesi; altrimenti restituisce un record che indica
    l'assenza di risultati.

    Args:
        session: sessione HTTP.
        sito: URL base del sito del comune (es. "https://www.comune.torino.it").
        comune: nome del comune.

    Returns:
        Lista di dizionari compatibili con ``progetti_master``.
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
    risultati: List[dict] = []
    for p in paths:
        url = urljoin(sito.rstrip("/") + "/", p.lstrip("/"))
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    continue
                html = await resp.text()
        except Exception:
            continue
        soup = BeautifulSoup(html, "html.parser")
        link_elems = soup.find_all("a", href=keywords_href)
        text_hits = soup.find_all(string=keywords_text)
        if not link_elems and not text_hits:
            continue
        titoli = [
            t.strip()
            for t in text_hits
            if isinstance(t, str) and len(t.strip()) > 10
        ][:10]
        risultati.append(
            {
                "timestamp": datetime.now().isoformat(),
                "fonte": "AT",
                "comune": comune,
                "identificativo": "",
                "titolo": f"Bandi/AT trovati su {url}",
                "importo": 0.0,
                "progettista": "",
                "data_evento": "",
                "url_dettaglio": url,
                "raw_data": json.dumps(
                    {"count_link": len(link_elems), "titoli": titoli}, ensure_ascii=False
                ),
            }
        )
    if not risultati:
        risultati.append(
            {
                "timestamp": datetime.now().isoformat(),
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


async def batch_at(siti_dict: Dict[str, str]) -> List[dict]:
    """Scraping AT per più comuni in parallelo."""
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [scrape_at_live(session, sito, comune) for comune, sito in siti_dict.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    flat: List[dict] = []
    for res in results:
        if isinstance(res, list):
            flat.extend(res)
    return flat


# ---------------------------------------------------------------------------
# Esempio di utilizzo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Connessione al database
    db = get_db()
    # 1. Scarica dati ANAC (primo mese 2025)
    anac_df = fetch_anac_cig_2025(nrows=200)
    if not anac_df.empty:
        anac_df.to_sql("progetti_master", db, if_exists="append", index=False)
        print(f"Inserite {len(anac_df)} righe ANAC.")
    else:
        print("Nessun dato ANAC recuperato.")
    # 2. OpenCUP – codici fiscali di esempio
    cfs = ["01206740324", "80012345678", "83001234567"]
    opencup_data = asyncio.run(batch_opencup(cfs))
    if opencup_data:
        pd.DataFrame(opencup_data).to_sql("progetti_master", db, if_exists="append", index=False)
        print(f"Inseriti {len(opencup_data)} record OpenCUP.")
    else:
        print("Nessun dato OpenCUP recuperato (verificare accesso open data/API).")
    # 3. OpenPNRR – comuni di esempio
    comuni_pnrr = ["Torino", "Milano", "Roma", "Napoli", "Firenze"]
    openpnrr_data = asyncio.run(batch_openpnrr(comuni_pnrr, max_results_per_comune=5))
    if openpnrr_data:
        pd.DataFrame(openpnrr_data).to_sql("progetti_master", db, if_exists="append", index=False)
        print(f"Inseriti {len(openpnrr_data)} record OpenPNRR.")
    else:
        print("Nessun dato OpenPNRR recuperato.")
    # 4. AT scraping
    siti = {
        "Torino": "https://www.comune.torino.it",
        "Milano": "https://www.comune.milano.it",
        "Roma": "https://www.comune.roma.it",
        "Napoli": "https://www.comune.napoli.it",
        "Firenze": "https://www.comune.fi.it",
    }
    at_data = asyncio.run(batch_at(siti))
    if at_data:
        pd.DataFrame(at_data).to_sql("progetti_master", db, if_exists="append", index=False)
        print(f"Inseriti {len(at_data)} record AT.")
    else:
        print("Nessun dato AT recuperato.")
    # Mostra un riepilogo
    df_master = pd.read_sql("SELECT fonte, count(*) as cnt FROM progetti_master GROUP BY fonte", db)
    print(df_master)
