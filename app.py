# app.py
# Ricerca Progetti Italia – Streamlit (lite + toggles)
# Avvio rapido su Streamlit Cloud, caricamenti opzionali, ricerca URL + dataset, progress/ETA.

from __future__ import annotations

import io
import re
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Ricerca Progetti Italia (Lite)", page_icon="🔎", layout="wide")

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

CATEGORIES: Dict[str, List[str]] = {
    "Sport e impianti sportivi (impianti, piscine, palestre, palazzetti)": [
        "impianto sportivo", "stadio", "palazzetto", "palestra", "piscina",
        "polisportivo", "campo", "centro sportivo",
    ],
    "Piste ciclabili e mobilità dolce (piste ciclabili, ciclovie, percorsi ciclo-pedonali)": [
        "pista ciclabile", "ciclovia", "ciclopedonale", "greenway",
        "mobilità dolce", "percorso ciclabile", "percorso ciclo pedonale",
    ],
    "Progetti aeroportuali": ["aeroporto", "aerostazione", "terminal", "hangar", "airside", "landside"],
    "Centri commerciali": ["centro commerciale", "mall", "galleria commerciale", "retail park", "ipermercato"],
    "Progetti alberghieri": ["hotel", "albergo", "resort", "struttura ricettiva", "ospitalità"],
    "Edilizia residenziale pubblica (ERP, housing, rigenerazione urbana, ristrutturazioni)": [
        "ERP", "edilizia residenziale pubblica", "housing", "social housing",
        "rigenerazione urbana", "ristrutturazione", "case popolari",
    ],
    "Strutture sanitarie (ospedali, poliambulatori, RSA, laboratori, fisioterapia)": [
        "ospedale", "poliambulatorio", "RSA", "laboratorio", "fisioterapia",
        "struttura sanitaria", "ambulatorio",
    ],
    "Università e ricerca (campus, laboratori, biblioteche, residenze)": [
        "campus", "laboratorio", "biblioteca", "residenza universitaria", "dipartimento",
    ],
    "Innovazione e startup (hub, incubatori, fab lab, maker space)": [
        "hub", "incubatore", "fab lab", "maker space", "acceleratore", "innovation",
    ],
    "Archivi e patrimonio culturale (archivi, biblioteche, musei, restauro)": [
        "archivio", "deposito", "biblioteca", "museo", "restauro", "patrimonio culturale",
    ],
    "Musei e spazi culturali (musei civici, gallerie, centri culturali)": [
        "museo", "musei civici", "galleria", "spazio culturale", "centro culturale",
    ],
    "Intrattenimento e spettacolo (teatri, cinema, auditorium, anfiteatri, spazi concerti)": [
        "teatro", "cinema", "auditorium", "anfiteatro", "spazio concerti", "arena",
    ],
    "Trasporto pubblico locale (tram, autobus, metro, bus elettrici, stazioni, bike sharing)": [
        "tram", "autobus", "metro", "bus elettrici", "stazione", "bike sharing", "deposito bus",
    ],
    "Edilizia giudiziaria e sicurezza (tribunali, carceri, questure, caserme)": [
        "tribunale", "carcere", "questura", "caserma", "comando",
    ],
    "Sistemazioni urbane (riqualificazione urbana, piazze, strade, parchi, arredo urbano)": [
        "riqualificazione urbana", "piazza", "strada", "parco", "arredo urbano", "sistemazione urbana",
    ],
}

PROJECT_STATUS = ["(qualsiasi)", "progetto PFTE", "progetto definitivo", "progetto esecutivo"]
PROJECT_PHASE = ["(qualsiasi)", "FASE DI PROGRAMMAZIONE", "FASE DI PROGETTAZIONE", "FASE DI ESECUZIONE"]

DATE_RE = re.compile(r"\b(?:(?:0?[1-9]|[12]\d|3[01])[-/\.](?:0?[1-9]|1[0-2])[-/\.](?:19|20)\d{2})\b")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)


# ----------------------------
# Models
# ----------------------------
@dataclass
class SearchResult:
    source_url: str
    found_url: str
    title: str
    matched_terms: List[str]
    doc_date: Optional[str] = None
    emails: Optional[List[str]] = None
    note: str = ""


# ----------------------------
# Network helpers
# ----------------------------
def safe_get(url: str, timeout: int = 25) -> requests.Response:
    headers = {"User-Agent": DEFAULT_UA, "Accept-Language": "it-IT,it;q=0.9,en;q=0.8"}
    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return resp


def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u


def guess_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())


def extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(base_url, href)
        if full.startswith(("mailto:", "javascript:")):
            continue
        links.append(full)
    # de-dup preserving order
    seen, out = set(), []
    for x in links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def is_probably_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


# ----------------------------
# Search helpers
# ----------------------------
def compile_terms(category: str, extra_terms: str, status: str, phase: str) -> List[str]:
    terms: List[str] = []
    if category and category != "(qualsiasi)":
        terms.extend(CATEGORIES.get(category, []))
    if status and status != "(qualsiasi)":
        terms.append(status)
    if phase and phase != "(qualsiasi)":
        terms.append(phase)
    if extra_terms:
        for t in re.split(r"[,\n;]+", extra_terms):
            t = t.strip()
            if t:
                terms.append(t)
    # unique (keep order)
    seen, cleaned = set(), []
    for t in terms:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            cleaned.append(t)
    return cleaned


def find_matches(text: str, terms: List[str]) -> List[str]:
    low = text.lower()
    return [t for t in terms if t.lower() in low]


def parse_date_from_text(text: str) -> Optional[str]:
    m = DATE_RE.search(text)
    return m.group(0) if m else None


def in_date_range(date_str: Optional[str], start, end) -> bool:
    """Filtri veri se la data è trovata; se non è trovata non escludiamo (ma resta DOC_DATE vuota)."""
    if not start and not end:
        return True
    if not date_str:
        return True
    d = None
    for sep in ("/", "-", "."):
        if sep in date_str:
            p = date_str.split(sep)
            if len(p) == 3:
                try:
                    import datetime as _dt
                    d = _dt.date(int(p[2]), int(p[1]), int(p[0]))
                except Exception:
                    d = None
            break
    if not d:
        return True
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True


def pdf_extract_text_and_meta(url: str, max_bytes: int = 12_000_000):
    """Usa pypdf SOLO se il toggle è ON e la lib è installata."""
    note = ""
    try:
        resp = safe_get(url, timeout=35)
        content = resp.content
        if len(content) > max_bytes:
            return "", None, [], f"PDF troppo grande ({len(content)/1e6:.1f} MB), skip parsing"

        from pypdf import PdfReader  # lazy import
        reader = PdfReader(io.BytesIO(content))

        parts = []
        for page in reader.pages[:6]:  # prime 6 pagine
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(parts)
        date_str = parse_date_from_text(text)
        emails = sorted(set(EMAIL_RE.findall(text)))
        return text, date_str, emails, note

    except ModuleNotFoundError:
        return "", None, [], "pypdf non installato"
    except Exception as e:
        return "", None, [], f"Errore parsing PDF: {e}"


def selenium_fetch_html(url: str, wait_s: int = 6) -> str:
    """Usa Selenium SOLO se toggle ON e selenium/chromium disponibili."""
    from selenium import webdriver  # lazy
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,1024")
    options.add_argument(f"--user-agent={DEFAULT_UA}")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(max(1, wait_s))
        return driver.page_source or ""
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def fetch_html(url: str, use_selenium: bool):
    try:
        if use_selenium:
            return selenium_fetch_html(url, wait_s=6), "selenium"
        r = safe_get(url, timeout=25)
        return r.text, "requests"
    except ModuleNotFoundError:
        return "", "selenium non installato"
    except Exception as e:
        return "", f"fetch error: {e}"


@st.cache_data(show_spinner=False)
def load_csv_robust_from_bytes(raw: bytes) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    try:
        df = pd.read_csv(io.BytesIO(raw))
        return df, warnings
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", on_bad_lines="skip")
        warnings.append("CSV letto in modalità robusta (sep auto + skip righe malformate).")
        return df, warnings


@st.cache_data(show_spinner=False)
def load_csv_robust(path_or_url: str) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if not path_or_url:
        return pd.DataFrame(), ["Percorso/URL CSV vuoto"]
    try:
        if path_or_url.startswith(("http://", "https://")):
            raw = safe_get(path_or_url, timeout=30).content
        else:
            with open(path_or_url, "rb") as f:
                raw = f.read()
        return load_csv_robust_from_bytes(raw)
    except Exception as e:
        return pd.DataFrame(), [f"Errore caricamento CSV: {e}"]


def show_working_indicator(step_text: str, progress: float, eta_s):
    cols = st.columns([2, 1])
    with cols[0]:
        st.info(f"⏳ In lavorazione: {step_text}")
        st.progress(min(max(progress, 0.0), 1.0))
    with cols[1]:
        st.metric("ETA stimata", f"{int(max(0, eta_s))} s" if eta_s is not None and math.isfinite(eta_s) else "—")


def crawl_url(
    start_url: str,
    terms: List[str],
    use_selenium: bool,
    parse_pdf_toggle: bool,
    start_date,
    end_date,
    max_results: int,
    max_depth: int,
    status_box=None
) -> List[SearchResult]:
    start_url = normalize_url(start_url)
    if not start_url:
        return []

    domain = urlparse(start_url).netloc
    visited = set()
    queue: List[Tuple[str, int]] = [(start_url, 0)]
    results: List[SearchResult] = []

    t0 = time.time()
    processed = 0
    rough_total = 1 + (max_depth * 25)

    while queue and len(results) < max_results:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        processed += 1

        # progress + ETA locale (per singolo URL)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, rough_total - processed)
        eta = remaining / rate if rate > 0 else None

        if status_box is not None:
            with status_box:
                show_working_indicator(f"{urlparse(url).netloc} · depth {depth}", processed / max(rough_total, 1), eta)

        # PDF
        if is_probably_pdf(url):
            title = url.split("/")[-1]
            matched_terms: List[str] = []
            doc_date, emails, note = None, [], ""

            if parse_pdf_toggle:
                text, doc_date, emails, note = pdf_extract_text_and_meta(url)
                if text:
                    matched_terms = find_matches(text, terms)

            if (not terms) or matched_terms:
                if in_date_range(doc_date, start_date, end_date):
                    results.append(SearchResult(start_url, url, title, matched_terms, doc_date, emails, note))
            continue

        # HTML
        html, fetch_note = fetch_html(url, use_selenium)
        if not html:
            continue

        text = guess_text_from_html(html)
        matched = find_matches(text, terms) if terms else []

        if (not terms) or matched:
            doc_date = parse_date_from_text(text)
            if in_date_range(doc_date, start_date, end_date):
                results.append(
                    SearchResult(
                        start_url,
                        url,
                        (text[:90] + "…") if len(text) > 90 else (text[:90] or url),
                        matched,
                        doc_date,
                        sorted(set(EMAIL_RE.findall(text)))[:20],
                        f"html({fetch_note})",
                    )
                )

        # Expand links (same domain only)
        if depth < max_depth and len(results) < max_results:
            links = extract_links(url, html)
            for lk in links[:150]:
                try:
                    if urlparse(lk).netloc and urlparse(lk).netloc != domain:
                        continue
                except Exception:
                    continue
                if lk not in visited:
                    queue.append((lk, depth + 1))

    return results


# ----------------------------
# UI
# ----------------------------
def sidebar_controls():
    st.sidebar.header("⚙️ Menu & filtri")
    nav = st.sidebar.radio("Menu", ["Home", "Dataset", "Ricerca"], index=0)

    st.sidebar.divider()
    st.sidebar.subheader("Opzioni (attivabili)")
    use_selenium = st.sidebar.toggle("Modalità Selenium", value=False, help="Usa Selenium solo se requests viene bloccato.")
    parse_pdf = st.sidebar.toggle("Parsing PDF (pypdf)", value=False, help="Scarica e analizza PDF: più lento.")

    st.sidebar.divider()
    st.sidebar.subheader("Tipi progetto")
    category = st.sidebar.selectbox("Categoria", ["(qualsiasi)"] + list(CATEGORIES.keys()), index=1)

    st.sidebar.subheader("Stato e fase")
    status = st.sidebar.selectbox("Stato del progetto", PROJECT_STATUS, index=0)
    phase = st.sidebar.selectbox("Fase", PROJECT_PHASE, index=0)

    st.sidebar.divider()
    st.sidebar.subheader("Filtro date (se disponibile)")
    c1, c2 = st.sidebar.columns(2)
    start = c1.date_input("data_inizio", value=None)
    end = c2.date_input("data_fine", value=None)

    st.sidebar.divider()
    st.sidebar.subheader("Prestazioni")
    max_results = st.sidebar.slider("Max risultati", 10, 250, 60, 10)
    max_depth = st.sidebar.slider("Profondità link HTML", 0, 2, 1, 1, help="0=solo pagina, 1=anche link, 2=più lento")

    return {
        "nav": nav,
        "use_selenium": use_selenium,
        "parse_pdf": parse_pdf,
        "category": category,
        "status": status,
        "phase": phase,
        "start_date": start if hasattr(start, "year") else None,
        "end_date": end if hasattr(end, "year") else None,
        "max_results": int(max_results),
        "max_depth": int(max_depth),
    }


def page_home():
    st.title("🔎 Ricerca Progetti Italia – Streamlit (Lite)")
    st.write(
        "Versione stabile per Streamlit Cloud: evita loop di caricamento e rende opzionali Selenium/PDF.\n\n"
        "Usa il menu a sinistra per scegliere cosa fare."
    )
    st.markdown(
        "- **Dataset**: carichi CSV (robusto, gestisce righe malformate)\n"
        "- **Ricerca**: incolli URL da scrappare e scegli categoria/stato/fase/date\n"
        "- Indicatore: mostra sempre che sta lavorando + ETA stimata\n"
    )


def page_dataset():
    st.header("📁 Dataset")
    st.write("Carica un CSV (file o URL). Parsing robusto: `sep auto` + `skip` righe malformate.")

    colA, colB = st.columns([1, 1])
    with colA:
        up = st.file_uploader("Upload CSV", type=["csv"])
    with colB:
        url = st.text_input("...oppure URL CSV", value="")

    df = None
    warnings: List[str] = []

    if up is not None:
        raw = up.read()
        df, warnings = load_csv_robust_from_bytes(raw)
    elif url.strip():
        df, warnings = load_csv_robust(url.strip())

    for w in warnings:
        st.warning(w)

    if df is None or df.empty:
        st.info("Nessun dataset caricato.")
        st.session_state.pop("dataset_df", None)
        st.session_state.pop("dataset_url_col", None)
        return

    st.session_state["dataset_df"] = df
    st.success(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    st.dataframe(df, use_container_width=True)

    url_cols = [c for c in df.columns if "url" in str(c).lower() or "link" in str(c).lower()]
    st.markdown("### Colonna URL (opzionale)")
    if url_cols:
        col = st.selectbox("Seleziona colonna URL", url_cols)
        st.session_state["dataset_url_col"] = col
        st.caption("In Ricerca potrai includere gli URL di questa colonna.")
    else:
        st.caption("Non ho trovato colonne che sembrano contenere URL (nomi con 'url' o 'link').")


def page_search(state):
    st.header("🧭 Ricerca")

    st.markdown("### Sorgenti URL")
    col1, col2 = st.columns([1, 1])

    with col1:
        # ✅ richiesta: possibilità di inserire un URL da scrappare
        urls_text = st.text_area(
            "Incolla uno o più URL (uno per riga)",
            value="",
            height=120,
            placeholder="https://www.comune.esempio.it/albo-pretorio\nhttps://www....",
        )
        extra_terms = st.text_area(
            "Keyword extra (comma/righe)",
            value="",
            height=80,
            placeholder='es: "affidamento progettazione", "incarico", CUP, CIG',
        )

    with col2:
        use_dataset = st.checkbox("Usa anche URL dal dataset caricato", value=False)
        limit_urls = st.number_input("Max URL dal dataset", 1, 5000, 200, 50)

        dataset_info = "—"
        if use_dataset and "dataset_df" in st.session_state:
            df = st.session_state["dataset_df"]
            col = st.session_state.get("dataset_url_col")
            if col and col in df.columns:
                dataset_info = f"{len(df[col].dropna().unique())} URL unici dalla colonna '{col}'"
            else:
                dataset_info = "Dataset caricato ma colonna URL non selezionata."
        st.write(dataset_info)

    # Build URL list
    urls: List[str] = []
    if urls_text.strip():
        for line in urls_text.splitlines():
            u = normalize_url(line)
            if u:
                urls.append(u)

    if use_dataset and "dataset_df" in st.session_state:
        df = st.session_state["dataset_df"]
        col = st.session_state.get("dataset_url_col")
        if col and col in df.columns:
            from_ds = [normalize_url(x) for x in df[col].dropna().astype(str).tolist()]
            from_ds = [x for x in from_ds if x]
            seen = set(urls)
            for u in from_ds:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
            urls = urls[: int(limit_urls)]
        else:
            st.warning("Dataset attivo ma colonna URL non selezionata in Dataset.")

    if not urls:
        st.info("Inserisci almeno 1 URL (o carica un dataset e seleziona la colonna URL).")
        return

    terms = compile_terms(state["category"], extra_terms, state["status"], state["phase"])

    st.markdown("### Parametri attivi")
    st.write(
        {
            "urls": len(urls),
            "categoria": state["category"],
            "stato": state["status"],
            "fase": state["phase"],
            "terms (preview)": terms[:25],
            "selenium": state["use_selenium"],
            "parse_pdf": state["parse_pdf"],
            "data_inizio": str(state["start_date"]) if state["start_date"] else None,
            "data_fine": str(state["end_date"]) if state["end_date"] else None,
            "depth": state["max_depth"],
        }
    )

    if not st.button("🚀 Avvia ricerca", type="primary"):
        st.caption("Suggerimento: lascia Selenium e parsing PDF OFF finché non ti servono.")
        return

    # ✅ soluzione A: indicator che sta lavorando + ETA
    status_box = st.container()
    results_all: List[SearchResult] = []
    t0 = time.time()
    total_urls = len(urls)
    hard_max = int(state["max_results"])

    for i, u in enumerate(urls, start=1):
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0.0
        remaining_urls = total_urls - i
        eta = (remaining_urls / rate) if rate > 0 else None
        show_working_indicator(f"URL {i}/{total_urls}: {u}", i / max(total_urls, 1), eta)

        try:
            chunk = crawl_url(
                start_url=u,
                terms=terms,
                use_selenium=state["use_selenium"],
                parse_pdf_toggle=state["parse_pdf"],
                start_date=state["start_date"],
                end_date=state["end_date"],
                max_results=max(1, hard_max - len(results_all)),
                max_depth=int(state["max_depth"]),
                status_box=status_box,
            )
            results_all.extend(chunk)
        except Exception as e:
            results_all.append(SearchResult(u, u, "ERRORE", [], note=f"Errore ricerca: {e}"))

        if len(results_all) >= hard_max:
            break

    st.success(f"Ricerca completata: {len(results_all)} risultati (limite {hard_max}).")

    out_df = pd.DataFrame(
        [
            {
                "SOURCE_URL": r.source_url,
                "FOUND_URL": r.found_url,
                "TITLE": r.title,
                "MATCHED_TERMS": ", ".join(r.matched_terms),
                "DOC_DATE": r.doc_date or "",
                "EMAILS": ", ".join(r.emails or []),
                "NOTE": r.note or "",
            }
            for r in results_all
        ]
    )

    st.dataframe(out_df, use_container_width=True)
    st.download_button(
        "⬇️ Scarica risultati CSV",
        out_df.to_csv(index=False).encode("utf-8"),
        file_name="risultati_ricerca.csv",
        mime="text/csv",
    )


def main():
    state = sidebar_controls()
    if state["nav"] == "Home":
        page_home()
    elif state["nav"] == "Dataset":
        page_dataset()
    else:
        page_search(state)


if __name__ == "__main__":
    main()
