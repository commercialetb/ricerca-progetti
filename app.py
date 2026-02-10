# app.py — Streamlit Cloud safe:
# - Lazy loading
# - Light scraping (requests/bs4)
# - Optional Selenium mode (toggle)
# - Optional PDF parsing w/ pypdf (toggle)
# - True date filters (from PDF when available)
# - Dropdown filters for category / status / phase
# - Progress + ETA

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
APP_TITLE = "Ricerca Progetti"
DEFAULT_TIMEOUT = 20

PDF_RE = re.compile(r"\.pdf(\?|$)", re.IGNORECASE)

# Date patterns in Italian docs
DATE_PATTERNS = [
    re.compile(r"\b(\d{2})[\/\-.](\d{2})[\/\-.](\d{4})\b"),  # dd/mm/yyyy
    re.compile(r"\b(\d{4})[\/\-.](\d{2})[\/\-.](\d{2})\b"),  # yyyy-mm-dd
]

# Limit PDF download to avoid timeouts / huge files
PDF_MAX_MB_DEFAULT = 10

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")


# -----------------------------
# Taxonomy (dropdown -> keywords)
# -----------------------------
CATEGORIES = [
    "Tutte",
    "Sport e impianti sportivi",
    "Piste ciclabili e mobilità dolce",
    "Progetti aeroportuali",
    "Centri commerciali",
    "Progetti alberghieri",
    "Edilizia residenziale pubblica",
    "Strutture sanitarie",
    "Università e ricerca",
    "Innovazione e startup",
    "Archivi e patrimonio culturale",
    "Musei e spazi culturali",
    "Intrattenimento e spettacolo",
    "Trasporto pubblico locale",
    "Edilizia giudiziaria e sicurezza",
    "Sistemazioni urbane",
]

CATEGORY_KEYWORDS = {
    "Sport e impianti sportivi": [
        "impianto sportivo", "palazzetto", "palestra", "piscina", "stadio", "campo sportivo", "palasport"
    ],
    "Piste ciclabili e mobilità dolce": [
        "pista ciclabile", "ciclovia", "percorso ciclopedonale", "ciclo-pedonale", "mobilità dolce", "bike sharing"
    ],
    "Progetti aeroportuali": ["aeroporto", "aerostazione", "terminal", "pista di volo", "airside", "landside"],
    "Centri commerciali": ["centro commerciale", "mall", "galleria commerciale", "retail park", "ipermercato"],
    "Progetti alberghieri": ["hotel", "albergo", "resort", "ospitalità", "ricettivo", "struttura ricettiva"],
    "Edilizia residenziale pubblica": ["erp", "edilizia residenziale pubblica", "housing", "rigenerazione urbana", "ristrutturazione alloggi"],
    "Strutture sanitarie": ["ospedale", "poliambulatorio", "rsa", "residenza sanitaria", "laboratorio", "fisioterapia", "ambulatorio"],
    "Università e ricerca": ["università", "campus", "laboratorio", "biblioteca", "residenza universitaria", "ricerca"],
    "Innovazione e startup": ["hub", "incubatore", "fab lab", "fablab", "maker space", "makerspace", "coworking", "startup"],
    "Archivi e patrimonio culturale": ["archivio", "restauro", "patrimonio culturale", "biblioteca", "catalogazione"],
    "Musei e spazi culturali": ["museo", "galleria", "centro culturale", "spazio culturale", "mostra"],
    "Intrattenimento e spettacolo": ["teatro", "cinema", "auditorium", "anfiteatro", "concerti", "spazio eventi"],
    "Trasporto pubblico locale": ["tram", "metro", "metropolitana", "autobus", "bus elettrici", "stazione", "deposito", "tpl"],
    "Edilizia giudiziaria e sicurezza": ["tribunale", "carcere", "questura", "caserma", "sicurezza", "prefettura"],
    "Sistemazioni urbane": ["riqualificazione urbana", "piazza", "strada", "parco", "arredo urbano", "sistemazione urbana"],
}

STATI = ["Tutti", "progetto PFTE", "progetto definitivo", "progetto esecutivo"]
STATO_KEYWORDS = {
    "progetto PFTE": ["pfte", "fattibilità tecnico-economica", "fattibilita tecnico economica", "studio di fattibilità"],
    "progetto definitivo": ["progetto definitivo"],
    "progetto esecutivo": ["progetto esecutivo"],
}

FASI = ["Tutte", "FASE DI PROGRAMMAZIONE", "FASE DI PROGETTAZIONE", "FASE DI ESECUZIONE"]
FASE_KEYWORDS = {
    "FASE DI PROGRAMMAZIONE": ["programmazione", "piano triennale", "programma", "inserimento in elenco", "cup", "opere pubbliche"],
    "FASE DI PROGETTAZIONE": ["progettazione", "affidamento progettazione", "incarico progettazione", "progetto", "pfte", "definitivo", "esecutivo"],
    "FASE DI ESECUZIONE": ["esecuzione", "lavori", "cantiere", "direzione lavori", "consegna lavori", "stato avanzamento", "sal"],
}


# -----------------------------
# Helpers
# -----------------------------
def seconds_to_hhmmss(s: float) -> str:
    s = max(0, int(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _safe_read_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], List[str]]:
    warnings = []
    try:
        df = pd.read_csv(
            uploaded_file,
            engine="python",
            sep=None,
            on_bad_lines="warn",
            dtype=str,
        )
        df.columns = [c.strip() for c in df.columns]
        return df, warnings
    except Exception as e:
        return None, [f"Errore lettura CSV: {e}"]


def normalize_targets_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings = []
    df = df.copy()

    url_col = None
    for candidate in ["ALBO_PRETORIO_URL", "URL", "PORTAL_URL", "LINK"]:
        if candidate in df.columns:
            url_col = candidate
            break

    if not url_col:
        return df, ["CSV senza colonna URL. Serve ALBO_PRETORIO_URL oppure URL."]

    if url_col != "ALBO_PRETORIO_URL":
        df.rename(columns={url_col: "ALBO_PRETORIO_URL"}, inplace=True)

    for c in ["COMUNE", "PROVINCIA", "REGIONE"]:
        if c not in df.columns:
            df[c] = ""

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


def extract_pdf_items_from_html(base_url: str, html: str, max_items: int = 400) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = urljoin(base_url, href)
        if not PDF_RE.search(full):
            continue
        anchor_text = " ".join((a.get_text(" ", strip=True) or "").split())
        # context snippet: try parent text
        parent_text = ""
        try:
            parent_text = " ".join((a.parent.get_text(" ", strip=True) or "").split())
        except Exception:
            parent_text = ""
        out.append({
            "pdf_url": full,
            "anchor_text": anchor_text,
            "context": parent_text[:500]
        })
        if len(out) >= max_items:
            break

    # dedup by pdf_url preserving order
    seen = set()
    dedup = []
    for x in out:
        if x["pdf_url"] not in seen:
            seen.add(x["pdf_url"])
            dedup.append(x)
    return dedup


def keyword_match_score(text: str, keywords: List[str]) -> int:
    if not text:
        return 0
    t = text.lower()
    score = 0
    for kw in keywords:
        if kw.lower() in t:
            score += 1
    return score


def choose_filters_keywords(category: str, stato: str, fase: str) -> Tuple[List[str], List[str], List[str]]:
    cat_k = CATEGORY_KEYWORDS.get(category, []) if category and category != "Tutte" else []
    stato_k = STATO_KEYWORDS.get(stato, []) if stato and stato != "Tutti" else []
    fase_k = FASE_KEYWORDS.get(fase, []) if fase and fase != "Tutte" else []
    return cat_k, stato_k, fase_k


def parse_date_from_text(text: str) -> Optional[date]:
    if not text:
        return None
    # find all candidate dates
    candidates: List[date] = []
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            try:
                if len(m.groups()) == 3:
                    g = m.groups()
                    # dd/mm/yyyy
                    if len(g[0]) == 2 and len(g[2]) == 4 and pat.pattern.startswith(r"\b(\d{2})"):
                        d = date(int(g[2]), int(g[1]), int(g[0]))
                    else:
                        # yyyy-mm-dd
                        d = date(int(g[0]), int(g[1]), int(g[2]))
                    candidates.append(d)
            except Exception:
                continue
    if not candidates:
        return None
    # heuristic: pick the most recent (often publication date is recent)
    candidates.sort()
    return candidates[-1]


def download_pdf_bytes(url: str, timeout: int, max_mb: int) -> Tuple[Optional[bytes], str]:
    try:
        with requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StreamlitBot/1.0)"},
            stream=True,
            allow_redirects=True,
        ) as r:
            if r.status_code >= 400:
                return None, f"HTTP {r.status_code}"
            total = 0
            chunks = []
            limit = max_mb * 1024 * 1024
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                total += len(chunk)
                if total > limit:
                    return None, f"PDF oltre limite ({max_mb}MB)"
                chunks.append(chunk)
            return b"".join(chunks), ""
    except Exception as e:
        return None, str(e)


def parse_pdf_text_and_date(pdf_bytes: bytes) -> Tuple[str, Optional[date], str]:
    """
    Returns: (text_snippet, doc_date, error)
    """
    try:
        from pypdf import PdfReader  # lazy import
    except Exception as e:
        return "", None, f"pypdf non disponibile: {e}"

    try:
        reader = PdfReader(io_bytes := _BytesIO(pdf_bytes))
        meta_date = None

        # metadata
        try:
            md = reader.metadata
            # common fields: /CreationDate, /ModDate
            for k in ["/ModDate", "/CreationDate"]:
                v = getattr(md, k, None) if md else None
                if not v and md and k in md:
                    v = md.get(k)
                if isinstance(v, str) and len(v) >= 8:
                    # pdf date often like D:20240131120000
                    m = re.search(r"(\d{4})(\d{2})(\d{2})", v)
                    if m:
                        meta_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                        break
        except Exception:
            pass

        # extract text (first N pages)
        texts = []
        max_pages = min(3, len(reader.pages))
        for i in range(max_pages):
            try:
                t = reader.pages[i].extract_text() or ""
                if t.strip():
                    texts.append(t)
            except Exception:
                continue

        full_text = "\n".join(texts).strip()
        snippet = full_text[:4000]

        text_date = parse_date_from_text(full_text)

        # choose best: prefer text_date (often “data” in doc), else metadata
        doc_date = text_date or meta_date

        return snippet, doc_date, ""
    except Exception as e:
        return "", None, str(e)


class _BytesIO:
    """Tiny BytesIO replacement (no import) to keep surface minimal."""
    def __init__(self, b: bytes):
        self._b = b
        self._i = 0

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            n = len(self._b) - self._i
        out = self._b[self._i:self._i + n]
        self._i += n
        return out

    def seek(self, i: int, whence: int = 0):
        if whence == 0:
            self._i = i
        elif whence == 1:
            self._i += i
        else:
            self._i = len(self._b) + i

    def tell(self):
        return self._i


@dataclass
class PortalResult:
    comune: str
    provincia: str
    regione: str
    portal_url: str
    error: str
    items: List[Dict]  # pdf items + optional parsed data


def selenium_collect_pdf_items(url: str, timeout: int, max_items: int = 400) -> Tuple[List[Dict[str, str]], str]:
    """
    Selenium mode: loads page in headless chromium and extracts PDF links.
    Lazy import so it won't break app startup.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except Exception as e:
        return [], f"Selenium non disponibile: {e}"

    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(1.2)

        html = driver.page_source
        driver.quit()

        return extract_pdf_items_from_html(url, html, max_items=max_items), ""
    except Exception as e:
        try:
            driver.quit()
        except Exception:
            pass
        return [], str(e)


def run_scrape(
    targets: List[Dict[str, str]],
    max_pages_per_portal: int,
    max_pdf_per_portal: int,
    timeout: int,
    use_selenium: bool,
    parse_pdf: bool,
    pdf_max_mb: int,
    date_start: Optional[date],
    date_end: Optional[date],
    include_no_date: bool,
    category: str,
    stato: str,
    fase: str,
    progress_cb=None,
    stop_flag_cb=None,
) -> List[Dict]:
    results: List[Dict] = []
    t0 = time.time()
    per_portal_times = []

    cat_k, stato_k, fase_k = choose_filters_keywords(category, stato, fase)

    for i, t in enumerate(targets, start=1):
        if stop_flag_cb and stop_flag_cb():
            break

        portal_url = (t.get("ALBO_PRETORIO_URL") or "").strip()
        comune = (t.get("COMUNE") or "").strip()
        provincia = (t.get("PROVINCIA") or "").strip()
        regione = (t.get("REGIONE") or "").strip()

        portal_start = time.time()
        err = ""
        items: List[Dict] = []

        # Candidate pages (light pagination heuristic)
        candidate_urls = [portal_url]
        if max_pages_per_portal > 1:
            for p in range(2, max_pages_per_portal + 1):
                if "?" in portal_url:
                    candidate_urls.append(f"{portal_url}&page={p}")
                else:
                    candidate_urls.append(f"{portal_url}?page={p}")

        # Selenium: typically only first page (avoid heavy loops)
        if use_selenium:
            collected, e = selenium_collect_pdf_items(portal_url, timeout=timeout, max_items=max_pdf_per_portal)
            if e:
                err = e
            items = collected[:max_pdf_per_portal]
        else:
            all_items = []
            for u in candidate_urls:
                if stop_flag_cb and stop_flag_cb():
                    break
                html, e = fetch_html(u, timeout=timeout)
                if e:
                    err = e
                    continue
                all_items.extend(extract_pdf_items_from_html(u, html, max_items=max_pdf_per_portal))

                if len(all_items) >= max_pdf_per_portal:
                    break

            # dedup and cap
            seen = set()
            dedup = []
            for x in all_items:
                if x["pdf_url"] not in seen:
                    seen.add(x["pdf_url"])
                    dedup.append(x)
            items = dedup[:max_pdf_per_portal]

        # Apply keyword filters (pre-PDF parsing using anchor/context)
        filtered_items = []
        for it in items:
            blob = f"{it.get('anchor_text','')}\n{it.get('context','')}".strip()
            score_cat = keyword_match_score(blob, cat_k) if cat_k else 1
            score_stato = keyword_match_score(blob, stato_k) if stato_k else 1
            score_fase = keyword_match_score(blob, fase_k) if fase_k else 1

            if (cat_k and score_cat == 0) or (stato_k and score_stato == 0) or (fase_k and score_fase == 0):
                continue

            it["_score_cat"] = score_cat
            it["_score_stato"] = score_stato
            it["_score_fase"] = score_fase
            filtered_items.append(it)

        # Optional PDF parsing (adds: pdf_text_snippet, doc_date, pdf_error)
        final_items = []
        for it in filtered_items:
            if stop_flag_cb and stop_flag_cb():
                break

            if parse_pdf:
                pdf_bytes, dl_err = download_pdf_bytes(it["pdf_url"], timeout=timeout, max_mb=pdf_max_mb)
                if dl_err:
                    it["pdf_error"] = dl_err
                    it["pdf_text_snippet"] = ""
                    it["doc_date"] = None
                else:
                    snippet, doc_dt, pe = parse_pdf_text_and_date(pdf_bytes)
                    it["pdf_text_snippet"] = snippet
                    it["doc_date"] = doc_dt
                    it["pdf_error"] = pe
            else:
                it["pdf_text_snippet"] = ""
                it["doc_date"] = None
                it["pdf_error"] = ""

            # Date filter (true) based on doc_date
            doc_dt = it.get("doc_date")
            if doc_dt is None:
                if not include_no_date and (date_start or date_end):
                    continue
            else:
                if date_start and doc_dt < date_start:
                    continue
                if date_end and doc_dt > date_end:
                    continue

            # Post-PDF keyword filter (more accurate if parsing enabled)
            if parse_pdf and (cat_k or stato_k or fase_k):
                text_blob = (it.get("pdf_text_snippet") or "").lower()
                if cat_k and keyword_match_score(text_blob, cat_k) == 0:
                    continue
                if stato_k and keyword_match_score(text_blob, stato_k) == 0:
                    continue
                if fase_k and keyword_match_score(text_blob, fase_k) == 0:
                    continue

            final_items.append(it)

        pr = PortalResult(
            comune=comune,
            provincia=provincia,
            regione=regione,
            portal_url=portal_url,
            error=err,
            items=final_items,
        )

        results.append({
            "comune": pr.comune,
            "provincia": pr.provincia,
            "regione": pr.regione,
            "portal_url": pr.portal_url,
            "error": pr.error,
            "n_items": len(pr.items),
            "items": pr.items,
        })

        portal_time = time.time() - portal_start
        per_portal_times.append(portal_time)

        if progress_cb:
            elapsed = time.time() - t0
            avg = sum(per_portal_times) / max(1, len(per_portal_times))
            remaining = (len(targets) - i) * avg
            progress_cb(i, len(targets), elapsed, remaining, pr)

    return results


# -----------------------------
# Session state
# -----------------------------
if "targets_df" not in st.session_state:
    st.session_state["targets_df"] = None
if "raw_results" not in st.session_state:
    st.session_state["raw_results"] = None
if "stop" not in st.session_state:
    st.session_state["stop"] = False

# feature toggles (default off to avoid heavy startup)
st.session_state.setdefault("enable_selenium_mode", False)
st.session_state.setdefault("enable_pdf_parsing", False)


# -----------------------------
# Sidebar + Router
# -----------------------------
st.title(APP_TITLE)
st.caption("Avvio stabile su Streamlit Cloud: carica e avvia solo ciò che selezioni dal menu.")

page = st.sidebar.radio(
    "📌 Menu",
    ["📂 Dataset", "🕷️ Ricerca", "📊 Risultati", "⚙️ Impostazioni"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.write("Stato:")
if st.session_state.get("raw_results") is not None:
    st.sidebar.success(f"Risultati presenti: {len(st.session_state['raw_results'])} portali")
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
            df = pd.DataFrame(columns=["COMUNE", "PROVINCIA", "REGIONE", "ALBO_PRETORIO_URL"])
        df = add_custom_url(df, custom_url)
        st.session_state["targets_df"] = df
        st.success("URL aggiunto (se non duplicato).")
        st.dataframe(df.tail(10), use_container_width=True)


def page_settings():
    st.subheader("⚙️ Impostazioni / Feature toggle")

    st.markdown("### Modalità avanzate (attivale solo quando ti servono)")
    st.session_state["enable_selenium_mode"] = st.checkbox(
        "Abilita Modalità Selenium (headless chromium)",
        value=st.session_state.get("enable_selenium_mode", False),
        help="Più potente su siti con JS, ma più lenta/pesante."
    )
    st.session_state["enable_pdf_parsing"] = st.checkbox(
        "Abilita Parsing PDF (pypdf)",
        value=st.session_state.get("enable_pdf_parsing", False),
        help="Scarica PDF e cerca testo + data nel documento. Più lento ma permette filtri data veri."
    )

    st.info(
        "Consiglio Cloud: lascia OFF Selenium e PDF parsing per test rapidi.\n"
        "Poi abilitali quando devi filtrare davvero per data/categoria."
    )


def page_search():
    st.subheader("🕷️ Ricerca")
    df = st.session_state.get("targets_df")
    if df is None or len(df) == 0:
        st.warning("Prima carica un dataset nella sezione 📂 Dataset.")
        return

    enable_selenium = bool(st.session_state.get("enable_selenium_mode"))
    enable_pdf = bool(st.session_state.get("enable_pdf_parsing"))

    # Filters
    st.markdown("### 🔽 Filtri progetto")
    c1, c2, c3 = st.columns(3)
    with c1:
        category = st.selectbox("Categoria progetto", CATEGORIES, index=0)
    with c2:
        stato = st.selectbox("Stato progetto", STATI, index=0)
    with c3:
        fase = st.selectbox("Fase", FASI, index=0)

    st.markdown("### 📅 Filtro date (basato su data estratta dal PDF quando disponibile)")
    d1, d2, d3 = st.columns([1, 1, 1])
    with d1:
        date_start = st.date_input("Data inizio", value=None)
    with d2:
        date_end = st.date_input("Data fine", value=None)
    with d3:
        include_no_date = st.checkbox("Includi documenti senza data trovata", value=True)

    # Params
    with st.expander("⚙️ Parametri ricerca", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            max_portals = st.number_input("Max portali", min_value=1, max_value=int(len(df)), value=min(30, int(len(df))))
        with col2:
            max_pages = st.number_input("Max pagine/portale (solo Light)", min_value=1, max_value=20, value=3, help="Con Selenium si usa solo la prima pagina.")
        with col3:
            max_pdf = st.number_input("Max PDF/portale", min_value=1, max_value=500, value=80)
        with col4:
            timeout = st.number_input("Timeout (sec)", min_value=5, max_value=60, value=20)
        with col5:
            pdf_max_mb = st.number_input(
                "Max MB per PDF",
                min_value=1,
                max_value=50,
                value=PDF_MAX_MB_DEFAULT,
                help="Limite download per evitare PDF enormi."
            )

    st.markdown("### ▶️ Avvio")
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        start_btn = st.button("🚀 Avvia ricerca", type="primary", use_container_width=True)
    with colB:
        stop_btn = st.button("🛑 Stop", use_container_width=True)

    if stop_btn:
        st.session_state["stop"] = True
        st.warning("Stop richiesto: la ricerca si fermerà al prossimo step.")

    # Live UI
    progress = st.progress(0)
    status = st.empty()
    eta_box = st.empty()
    last_box = st.empty()

    def progress_cb(done, total, elapsed, remaining, last_pr: PortalResult):
        progress.progress(done / total)
        status.info(f"Processati {done}/{total} portali — elapsed {seconds_to_hhmmss(elapsed)}")
        eta_box.success(f"⏳ ETA stimata: {seconds_to_hhmmss(remaining)} (countdown stimato)")
        last_box.write(
            f"Ultimo portale: **{last_pr.portal_url}**  | risultati: **{len(last_pr.items)}**"
            + (f" | errore: `{last_pr.error}`" if last_pr.error else "")
        )

    def stop_flag_cb():
        return bool(st.session_state.get("stop"))

    # Convert date inputs: Streamlit returns datetime.date or None
    ds = date_start if isinstance(date_start, date) else None
    de = date_end if isinstance(date_end, date) else None

    if start_btn:
        st.session_state["stop"] = False
        targets = df.head(int(max_portals)).to_dict("records")

        with st.spinner("Ricerca in corso… (progress/ETA sopra)"):
            results = run_scrape(
                targets=targets,
                max_pages_per_portal=int(max_pages),
                max_pdf_per_portal=int(max_pdf),
                timeout=int(timeout),
                use_selenium=enable_selenium,
                parse_pdf=enable_pdf,
                pdf_max_mb=int(pdf_max_mb),
                date_start=ds,
                date_end=de,
                include_no_date=bool(include_no_date),
                category=category,
                stato=stato,
                fase=fase,
                progress_cb=progress_cb,
                stop_flag_cb=stop_flag_cb,
            )

        st.session_state["raw_results"] = results
        st.success(f"Ricerca completata. Portali processati: {len(results)}")
        st.rerun()


def page_results():
    st.subheader("📊 Risultati")
    results = st.session_state.get("raw_results")
    if not results:
        st.info("Nessun risultato. Avvia una ricerca da 🕷️ Ricerca.")
        return

    # Flatten items
    flat = []
    for r in results:
        base = {
            "comune": r.get("comune"),
            "provincia": r.get("provincia"),
            "regione": r.get("regione"),
            "portal_url": r.get("portal_url"),
            "portal_error": r.get("error"),
        }
        for it in (r.get("items") or []):
            flat.append({
                **base,
                "pdf_url": it.get("pdf_url"),
                "anchor_text": it.get("anchor_text"),
                "context": it.get("context"),
                "doc_date": it.get("doc_date"),
                "pdf_error": it.get("pdf_error"),
            })

    df = pd.DataFrame(flat)
    st.write(f"Record estratti: **{len(df)}**")

    if len(df) > 0:
        # make dates readable
        if "doc_date" in df.columns:
            df["doc_date"] = df["doc_date"].astype(str).replace({"None": ""})

        st.dataframe(df, use_container_width=True, height=520)

        st.markdown("### 📎 Esporta")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Scarica risultati (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="scrape_results_flat.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            # portal summary
            summary = pd.DataFrame([{
                "comune": r.get("comune"),
                "provincia": r.get("provincia"),
                "regione": r.get("regione"),
                "portal_url": r.get("portal_url"),
                "portal_error": r.get("error"),
                "n_items": r.get("n_items"),
            } for r in results])
            st.download_button(
                "⬇️ Scarica riepilogo portali (CSV)",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="portals_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("### 🔎 Preview link (prime 200)")
        st.code("\n".join(df["pdf_url"].dropna().astype(str).tolist()[:200]) if len(df) else "—")
    else:
        st.warning("Nessun item trovato con i filtri correnti.")


# Router
if page == "📂 Dataset":
    page_dataset()
elif page == "🕷️ Ricerca":
    page_search()
elif page == "📊 Risultati":
    page_results()
else:
    page_settings()
