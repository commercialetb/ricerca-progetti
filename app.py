# app.py - Streamlit Cloud safe + menu + lazy loading + progress/ETA + custom URL scrape
import time
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "🧠 Ricerca Progetti — OSINT Agent (Streamlit Safe)"
DEFAULT_TIMEOUT = 20

PDF_RE = re.compile(r"\.pdf(\?|$)", re.IGNORECASE)

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def _safe_read_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Legge CSV in modo tollerante:
    - engine='python' per gestire righe "sporche"
    - on_bad_lines='warn' per non crashare
    """
    warnings = []
    try:
        df = pd.read_csv(
            uploaded_file,
            engine="python",
            sep=None,              # autodetect delimiter
            on_bad_lines="warn",   # non crashare su righe malformate
            dtype=str
        )
        df.columns = [c.strip() for c in df.columns]
        return df, warnings
    except Exception as e:
        return None, [f"Errore lettura CSV: {e}"]


def normalize_targets_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalizza il dataset portali: richiede almeno una colonna URL.
    Accetta ALBO_PRETORIO_URL oppure URL.
    """
    warnings = []
    df = df.copy()

    # prova a trovare colonna url
    url_col = None
    for candidate in ["ALBO_PRETORIO_URL", "URL", "PORTAL_URL", "LINK"]:
        if candidate in df.columns:
            url_col = candidate
            break

    if not url_col:
        return df, ["CSV senza colonna URL. Serve ALBO_PRETORIO_URL oppure URL."]

    # rinomina a standard
    if url_col != "ALBO_PRETORIO_URL":
        df.rename(columns={url_col: "ALBO_PRETORIO_URL"}, inplace=True)

    # colonne opzionali
    for c in ["COMUNE", "PROVINCIA", "REGIONE"]:
        if c not in df.columns:
            df[c] = ""

    # pulizia
    df["ALBO_PRETORIO_URL"] = df["ALBO_PRETORIO_URL"].fillna("").astype(str).str.strip()
    df = df[df["ALBO_PRETORIO_URL"] != ""].reset_index(drop=True)

    if len(df) == 0:
        warnings.append("Nessun URL valido trovato nel CSV.")

    return df, warnings


def add_custom_url(df: pd.DataFrame, url: str, comune: str = "", provincia: str = "", regione: str = "") -> pd.DataFrame:
    url = (url or "").strip()
    if not url:
        return df
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url

    # evita duplicati
    if "ALBO_PRETORIO_URL" in df.columns and (df["ALBO_PRETORIO_URL"] == url).any():
        return df

    new_row = {
        "COMUNE": comune or "CUSTOM",
        "PROVINCIA": provincia or "",
        "REGIONE": regione or "",
        "ALBO_PRETORIO_URL": url
    }
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[Optional[str], Optional[str]]:
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StreamlitBot/1.0)"},
            allow_redirects=True,
        )
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}"
        return r.text, None
    except Exception as e:
        return None, str(e)


def extract_pdf_links_from_html(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        if PDF_RE.search(full):
            links.append(full)
    # dedup preservando ordine
    seen = set()
    out = []
    for x in links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class ScrapeResult:
    comune: str
    provincia: str
    regione: str
    portal_url: str
    n_pdf: int
    pdf_links: List[str]
    error: str = ""


def run_light_scrape(
    targets: List[Dict[str, str]],
    max_pages_per_portal: int,
    max_pdf_per_portal: int,
    timeout: int,
    progress_cb=None,
    stop_flag_cb=None,
) -> List[Dict]:
    """
    Scraping "light":
    - prova a leggere la pagina del portale e a estrarre link pdf
    - se max_pages_per_portal > 1: tenta pattern semplici ?page=2 / &page=2
      (NON garantito, ma aiuta su alcuni albi)
    """
    results: List[Dict] = []
    t0 = time.time()
    per_portal_times = []

    for i, t in enumerate(targets, start=1):
        if stop_flag_cb and stop_flag_cb():
            break

        portal_url = (t.get("ALBO_PRETORIO_URL") or "").strip()
        comune = (t.get("COMUNE") or "").strip()
        provincia = (t.get("PROVINCIA") or "").strip()
        regione = (t.get("REGIONE") or "").strip()

        portal_start = time.time()

        all_pdfs = []
        err = ""

        # pagine: base + ?page=N / &page=N
        candidate_urls = [portal_url]
        if max_pages_per_portal > 1:
            for p in range(2, max_pages_per_portal + 1):
                if "?" in portal_url:
                    candidate_urls.append(f"{portal_url}&page={p}")
                else:
                    candidate_urls.append(f"{portal_url}?page={p}")

        for u in candidate_urls:
            if stop_flag_cb and stop_flag_cb():
                break
            html, e = fetch_html(u, timeout=timeout)
            if e:
                err = e
                continue
            pdfs = extract_pdf_links_from_html(u, html)
            all_pdfs.extend(pdfs)
            if len(all_pdfs) >= max_pdf_per_portal:
                break

        # dedup + cap
        seen = set()
        deduped = []
        for x in all_pdfs:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        deduped = deduped[:max_pdf_per_portal]

        sr = ScrapeResult(
            comune=comune,
            provincia=provincia,
            regione=regione,
            portal_url=portal_url,
            n_pdf=len(deduped),
            pdf_links=deduped,
            error=err,
        )
        results.append(sr.__dict__)

        portal_time = time.time() - portal_start
        per_portal_times.append(portal_time)

        if progress_cb:
            elapsed = time.time() - t0
            avg = sum(per_portal_times) / max(1, len(per_portal_times))
            remaining = (len(targets) - i) * avg
            progress_cb(i, len(targets), elapsed, remaining, sr)

    return results


def seconds_to_hhmmss(s: float) -> str:
    s = max(0, int(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


# -----------------------------
# UI State
# -----------------------------
if "targets_df" not in st.session_state:
    st.session_state["targets_df"] = None
if "raw_results" not in st.session_state:
    st.session_state["raw_results"] = None
if "stop" not in st.session_state:
    st.session_state["stop"] = False


# -----------------------------
# Sidebar Menu
# -----------------------------
st.title(APP_TITLE)
st.caption("Avvio rapido (no loop): carica e avvia solo ciò che selezioni dal menu.")

page = st.sidebar.radio(
    "📌 Menu",
    ["📂 Dataset", "🕷️ Scraping (Light)", "📊 Risultati", "⚙️ Impostazioni"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.write("Stato:")
if st.session_state.get("raw_results") is not None:
    st.sidebar.success(f"Risultati presenti: {len(st.session_state['raw_results'])} righe")
else:
    st.sidebar.info("Nessuna ricerca avviata")


# -----------------------------
# Pages
# -----------------------------
def page_dataset():
    st.subheader("📂 Dataset Portali")
    st.write("Carica un CSV con almeno una colonna URL (ALBO_PRETORIO_URL oppure URL).")

    uploaded = st.file_uploader("Carica CSV portali", type=["csv"])
    if uploaded:
        df, w1 = _safe_read_csv(uploaded)
        if w1:
            for w in w1:
                st.warning(w)

        if df is None:
            st.error("Impossibile leggere il CSV.")
            return

        df, w2 = normalize_targets_df(df)
        for w in w2:
            st.warning(w)

        st.session_state["targets_df"] = df
        st.success(f"Dataset caricato: {len(df)} portali validi")
        st.dataframe(df.head(30), use_container_width=True)

    st.markdown("### ➕ Aggiungi URL custom")
    colA, colB = st.columns([3, 1])
    with colA:
        custom_url = st.text_input("URL portale / albo da scrappare", placeholder="https://... oppure dominio.it/...")
    with colB:
        add_btn = st.button("Aggiungi", use_container_width=True)

    if add_btn:
        df = st.session_state.get("targets_df")
        if df is None:
            # crea DF base
            df = pd.DataFrame(columns=["COMUNE", "PROVINCIA", "REGIONE", "ALBO_PRETORIO_URL"])
        df = add_custom_url(df, custom_url)
        st.session_state["targets_df"] = df
        st.success("URL aggiunto (se non duplicato).")
        st.dataframe(df.tail(10), use_container_width=True)


def page_scraping_light():
    st.subheader("🕷️ Scraping (Light)")
    st.write("Modalità stabile per Streamlit Cloud: usa requests+BeautifulSoup per estrarre link PDF dalle pagine.")

    df = st.session_state.get("targets_df")
    if df is None or len(df) == 0:
        st.warning("Prima carica un dataset nella sezione 📂 Dataset.")
        return

    with st.expander("⚙️ Parametri ricerca", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_portals = st.number_input("Max portali da processare", min_value=1, max_value=int(len(df)), value=min(30, int(len(df))))
        with col2:
            max_pages = st.number_input("Max pagine/portale", min_value=1, max_value=20, value=3)
        with col3:
            max_pdf = st.number_input("Max PDF/portale", min_value=1, max_value=500, value=80)
        with col4:
            timeout = st.number_input("Timeout (sec)", min_value=5, max_value=60, value=20)

    st.markdown("### ▶️ Avvio")
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        start_btn = st.button("🚀 Avvia scraping", type="primary", use_container_width=True)
    with colB:
        stop_btn = st.button("🛑 Stop", use_container_width=True)

    if stop_btn:
        st.session_state["stop"] = True
        st.warning("Stop richiesto: la ricerca si fermerà al prossimo step.")

    # UI placeholders (live)
    progress = st.progress(0)
    status = st.empty()
    eta_box = st.empty()
    last_box = st.empty()

    def progress_cb(done, total, elapsed, remaining, last_sr: ScrapeResult):
        progress.progress(done / total)
        status.info(f"Processati {done}/{total} portali — elapsed {seconds_to_hhmmss(elapsed)}")
        eta_box.success(f"⏳ ETA stimata: {seconds_to_hhmmss(remaining)} (countdown stimato)")
        last_box.write(
            f"Ultimo portale: **{last_sr.portal_url}**  | PDF trovati: **{last_sr.n_pdf}**"
            + (f" | Errore: `{last_sr.error}`" if last_sr.error else "")
        )

    def stop_flag_cb():
        return bool(st.session_state.get("stop"))

    if start_btn:
        st.session_state["stop"] = False
        targets = df.head(int(max_portals)).to_dict("records")

        with st.spinner("Scraping in corso… (vedi progress/ETA sopra)"):
            results = run_light_scrape(
                targets=targets,
                max_pages_per_portal=int(max_pages),
                max_pdf_per_portal=int(max_pdf),
                timeout=int(timeout),
                progress_cb=progress_cb,
                stop_flag_cb=stop_flag_cb
            )

        st.session_state["raw_results"] = results
        st.success(f"Scraping completato. Risultati: {len(results)} righe")
        st.rerun()


def page_results():
    st.subheader("📊 Risultati")

    results = st.session_state.get("raw_results")
    if not results:
        st.info("Nessun risultato. Avvia una ricerca da 🕷️ Scraping (Light).")
        return

    df = pd.DataFrame(results)
    st.dataframe(df[["comune", "provincia", "regione", "portal_url", "n_pdf", "error"]], use_container_width=True, height=420)

    st.markdown("### 📎 Esporta")
    col1, col2 = st.columns(2)

    with col1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Scarica risultati (CSV)", data=csv_data, file_name="scrape_results.csv", mime="text/csv", use_container_width=True)

    with col2:
        # estrai lista unica pdf
        all_pdfs = []
        for row in results:
            for p in (row.get("pdf_links") or []):
                all_pdfs.append({"portal_url": row.get("portal_url"), "pdf_url": p})
        pdf_df = pd.DataFrame(all_pdfs)
        pdf_csv = pdf_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Scarica lista PDF (CSV)", data=pdf_csv, file_name="pdf_links.csv", mime="text/csv", use_container_width=True)

    st.markdown("### 🔎 Preview PDF links (prime 200)")
    all_flat = []
    for row in results:
        for p in (row.get("pdf_links") or []):
            all_flat.append(p)
    st.write(f"Totale link PDF estratti: **{len(all_flat)}**")
    st.code("\n".join(all_flat[:200]) if all_flat else "—")


def page_settings():
    st.subheader("⚙️ Impostazioni / Feature toggle")
    st.write("Qui puoi decidere cosa abilitare **dal frontend** senza appesantire l’avvio.")

    st.markdown("### Toggle (solo UI)")
    st.session_state["enable_pdf_parsing"] = st.checkbox("Abilita parsing PDF (non implementato qui)", value=st.session_state.get("enable_pdf_parsing", False))
    st.session_state["enable_ai_enrichment"] = st.checkbox("Abilita AI enrichment (non implementato qui)", value=st.session_state.get("enable_ai_enrichment", False))
    st.session_state["enable_selenium_mode"] = st.checkbox("Abilita Selenium mode (step successivo)", value=st.session_state.get("enable_selenium_mode", False))

    st.info(
        "Questi toggle non caricano librerie pesanti finché non implementiamo la modalità avanzata.\n\n"
        "Così la app **non va in loop** e parte sempre."
    )


# Router
if page == "📂 Dataset":
    page_dataset()
elif page == "🕷️ Scraping (Light)":
    page_scraping_light()
elif page == "📊 Risultati":
    page_results()
else:
    page_settings()
