# app.py
# Streamlit app: Ricerca Progetti (lightweight + toggles per moduli pesanti)
# Compatible with Streamlit Cloud (no background loops, lazy imports, safe fallbacks)

from __future__ import annotations

import io
import re
import time
import math
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Constants / Taxonomy
# -----------------------------
REGIONI = [
    "(tutte)",
    "Abruzzo","Basilicata","Calabria","Campania","Emilia-Romagna","Friuli-Venezia Giulia",
    "Lazio","Liguria","Lombardia","Marche","Molise","Piemonte","Puglia","Sardegna",
    "Sicilia","Toscana","Trentino-Alto Adige","Umbria","Valle d'Aosta","Veneto"
]

CATEGORIE = [
    "(tutte)",
    "Sport e impianti sportivi (impianti, piscine, palestre, palazzetti)",
    "Piste ciclabili e mobilità dolce (piste ciclabili, ciclovie, percorsi ciclo-pedonali)",
    "Progetti aeroportuali",
    "Centri commerciali",
    "Progetti alberghieri",
    "Edilizia residenziale pubblica (ERP, housing, rigenerazione urbana, ristrutturazioni)",
    "Strutture sanitarie (ospedali, poliambulatori, RSA, laboratori, fisioterapia)",
    "Università e ricerca (campus, laboratori, biblioteche, residenze)",
    "Innovazione e startup (hub, incubatori, fab lab, maker space)",
    "Archivi e patrimonio culturale (archivi, biblioteche, musei, restauro)",
    "Musei e spazi culturali (musei civici, gallerie, centri culturali)",
    "Intrattenimento e spettacolo (teatri, cinema, auditorium, anfiteatri, spazi concerti)",
    "Trasporto pubblico locale (tram, autobus, metro, bus elettrici, stazioni, bike sharing)",
    "Edilizia giudiziaria e sicurezza (tribunali, carceri, questure, caserme)",
    "Sistemazioni urbane (riqualificazione urbana, piazze, strade, parchi, arredo urbano)",
]

STATI_PROGETTO = ["(tutti)", "PFTE", "Definitivo", "Esecutivo", "N/D"]
FASE = ["(tutte)", "Programmazione", "Progettazione", "Esecuzione", "N/D"]

# Output columns requested (labels kept exact)
OUTPUT_COLUMNS = [
    "Data pubblicazione (DD/MM/YYYY)",
    "REGIONE (coerente con CIG/CUP)",
    "Comune/Città",
    "Nome Progetto",
    "Settore/Tipologia",
    "Ente Committente (Comune, Regione, ASL, Università, Privato, Pubblico-Privato)",
    "Tipo Ente (Pubblico, Pubblico-Privato, Privato)",
    "Valore Progetto Totale (€)",
    "Valore Fase Corrente (€)",
    "% Completamento (Preliminare, Definitiva, Esecutiva, DL, Collaudo)",
    "Data scadenza (DD/MM/YYYY)",
    "CIG",
    "CUP",
    "Portale/Fonte (OpenPNRR, OpenCUP, MEPA, GURI, Albo Pretorio, etc.)",
    "Studio/Progettista",
    "Fase Progettazione (Preliminare, Definitiva, Esecutiva, Direttore Lavori, Collaudatore)",
    # extra requested fields
    "Stato del progetto (PFTE/Definitivo/Esecutivo/N/D)",
    "Fase (Programmazione/Progettazione/Esecuzione/N/D)",
    "Settore (Terziario/Servizi/Residenziale/Industria/Ingegneria/N/D)",
    "Dettagli",
    "Progettazione Capogruppo",
    "Progettazione Direzione Lavori",
    "Progettazione Architettonica",
    "Studio di Fattibilità",
    "Responsabile Progetto",
    "Progetto Impianti Elettrici",
]

# -----------------------------
# Utilities
# -----------------------------
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b"),  # dd/mm/yyyy
    re.compile(r"\b(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})\b"),    # yyyy-mm-dd
]
CIG_RE = re.compile(r"\b([A-Z0-9]{10})\b")
CUP_RE = re.compile(r"\b([A-Z][A-Z0-9]{9,14})\b")
EURO_RE = re.compile(r"€\s?([0-9\.\,]+)")
VAL_RE = re.compile(r"\b([0-9]{1,3}(?:\.[0-9]{3})*(?:,[0-9]{2})?)\s?€\b")

def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    parsed = urlparse(u)
    return parsed._replace(fragment="").geturl()

def safe_get(url: str, timeout: int = 25, headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
    headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; StreamlitOSINT/1.0)"}
    try:
        resp = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        if resp.status_code >= 400:
            return None
        return resp
    except Exception:
        return None

def detect_region_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "regione" in str(c).lower():
            return c
    return None

def detect_url_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = str(c).lower()
        if "url" in s or "link" in s or "sito" in s:
            cols.append(c)
    return cols

def parse_dates(text: str) -> List[dt.date]:
    out: List[dt.date] = []
    t = text or ""
    for pat in DATE_PATTERNS:
        for m in pat.finditer(t):
            try:
                if len(m.groups()) == 3 and len(m.group(1)) == 4:
                    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                else:
                    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    if y < 100:
                        y += 2000
                out.append(dt.date(y, mo, d))
            except Exception:
                pass
    return out

def ddmmyyyy(d: Optional[dt.date]) -> str:
    return d.strftime("%d/%m/%Y") if d else "N/D"

def to_euro_str(val: Optional[float]) -> str:
    if val is None:
        return "N/D"
    s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"

def parse_first_euro(text: str) -> Optional[float]:
    if not text:
        return None
    m = EURO_RE.search(text) or VAL_RE.search(text)
    if not m:
        return None
    num = m.group(1)
    try:
        num = num.replace(".", "").replace(",", ".")
        return float(num)
    except Exception:
        return None

def host_label(url: str) -> str:
    try:
        h = urlparse(url).netloc.lower()
        return h or "N/D"
    except Exception:
        return "N/D"

def compile_terms(category: str, extra_terms: str, status: str, phase: str, region: str) -> List[str]:
    terms: List[str] = []
    if category and category != "(tutte)":
        terms.append(category.split("(")[0].strip())
        if "(" in category and ")" in category:
            inside = category.split("(", 1)[1].rsplit(")", 1)[0]
            terms += [t.strip() for t in inside.split(",") if t.strip()]
    if status and status != "(tutti)":
        terms.append(status)
    if phase and phase != "(tutte)":
        terms.append(phase)
    if region and region != "(tutte)":
        terms.append(region)

    if extra_terms:
        for x in re.split(r"[\n,;]+", extra_terms):
            x = x.strip().strip('"').strip("'")
            if x:
                terms.append(x)

    baseline = ["affidamento", "incarico", "progettazione", "PFTE", "definitivo", "esecutivo",
                "direzione lavori", "gara", "bando", "avviso", "CIG", "CUP"]
    terms += baseline

    seen = set()
    out = []
    for t in terms:
        t = re.sub(r"\s+", " ", str(t)).strip()
        if not t:
            continue
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out

def text_matches_terms(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    hits = 0
    for term in terms:
        if term.lower() in t:
            hits += 1
            if hits >= 2:
                return True
    return hits >= 1

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# -----------------------------
# Optional modules (lazy)
# -----------------------------
def selenium_fetch(url: str, timeout: int = 35) -> Optional[str]:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1280,800")

        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html = driver.page_source
        driver.quit()
        return html
    except Exception:
        return None

def pypdf_extract_text(pdf_bytes: bytes, max_pages: int = 10) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages[:max_pages]:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(texts)
    except Exception:
        return ""

def reportlab_pdf(df: pd.DataFrame, summary: str) -> Optional[bytes]:
    # PDF è opzionale: funziona solo se aggiungi reportlab a requirements.txt
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm

        buff = io.BytesIO()
        c = canvas.Canvas(buff, pagesize=landscape(A4))
        w, h = landscape(A4)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(1.5*cm, h-1.5*cm, "Report Ricerca Progetti")
        c.setFont("Helvetica", 10)
        y = h-2.3*cm
        for line in summary.splitlines():
            c.drawString(1.5*cm, y, line[:170])
            y -= 0.55*cm
        c.showPage()

        cols = list(df.columns)
        x0 = 1.0*cm
        row_h = 0.45*cm
        y = h-1.2*cm

        def header():
            nonlocal y
            c.setFont("Helvetica-Bold", 7)
            x = x0
            for col in cols:
                c.drawString(x, y, str(col)[:28])
                x += (w-2.0*cm)/len(cols)
            y -= 0.6*cm
            c.setFont("Helvetica", 6)

        header()
        for _, r in df.iterrows():
            if y < 1.2*cm:
                c.showPage()
                y = h-1.2*cm
                header()
            x = x0
            for col in cols:
                v = str(r.get(col, "")).replace("\n", " ")
                c.drawString(x, y, v[:28])
                x += (w-2.0*cm)/len(cols)
            y -= row_h

        c.save()
        buff.seek(0)
        return buff.read()
    except Exception:
        return None

# -----------------------------
# Data model
# -----------------------------
@dataclass
class ProjectRecord:
    pubblicazione: str = "N/D"
    regione: str = "N/D"
    comune: str = "N/D"
    nome_progetto: str = "N/D"
    tipologia: str = "N/D"
    committente: str = "N/D"
    tipo_ente: str = "N/D"
    valore_totale: str = "N/D"
    valore_fase: str = "N/D"
    percento: str = "N/D"
    scadenza: str = "N/D"
    cig: str = "N/D"
    cup: str = "N/D"
    fonte: str = "N/D"
    progettista: str = "N/D"
    fase_progettazione: str = "N/D"
    stato_progetto: str = "N/D"
    fase: str = "N/D"
    settore: str = "N/D"
    dettagli: str = "N/D"
    prog_capogruppo: str = "N/D"
    prog_dl: str = "N/D"
    prog_arch: str = "N/D"
    studio_fatt: str = "N/D"
    rup: str = "N/D"
    impianti_el: str = "N/D"

    def to_row(self) -> Dict[str, Any]:
        return {
            "Data pubblicazione (DD/MM/YYYY)": self.pubblicazione,
            "REGIONE (coerente con CIG/CUP)": self.regione,
            "Comune/Città": self.comune,
            "Nome Progetto": self.nome_progetto,
            "Settore/Tipologia": self.tipologia,
            "Ente Committente (Comune, Regione, ASL, Università, Privato, Pubblico-Privato)": self.committente,
            "Tipo Ente (Pubblico, Pubblico-Privato, Privato)": self.tipo_ente,
            "Valore Progetto Totale (€)": self.valore_totale,
            "Valore Fase Corrente (€)": self.valore_fase,
            "% Completamento (Preliminare, Definitiva, Esecutiva, DL, Collaudo)": self.percento,
            "Data scadenza (DD/MM/YYYY)": self.scadenza,
            "CIG": self.cig,
            "CUP": self.cup,
            "Portale/Fonte (OpenPNRR, OpenCUP, MEPA, GURI, Albo Pretorio, etc.)": self.fonte,
            "Studio/Progettista": self.progettista,
            "Fase Progettazione (Preliminare, Definitiva, Esecutiva, Direttore Lavori, Collaudatore)": self.fase_progettazione,
            "Stato del progetto (PFTE/Definitivo/Esecutivo/N/D)": self.stato_progetto,
            "Fase (Programmazione/Progettazione/Esecuzione/N/D)": self.fase,
            "Settore (Terziario/Servizi/Residenziale/Industria/Ingegneria/N/D)": self.settore,
            "Dettagli": self.dettagli,
            "Progettazione Capogruppo": self.prog_capogruppo,
            "Progettazione Direzione Lavori": self.prog_dl,
            "Progettazione Architettonica": self.prog_arch,
            "Studio di Fattibilità": self.studio_fatt,
            "Responsabile Progetto": self.rup,
            "Progetto Impianti Elettrici": self.impianti_el,
        }

# -----------------------------
# Extraction heuristics
# -----------------------------
def extract_from_text(text: str, url: str, state: Dict[str, Any]) -> ProjectRecord:
    t = (text or "").replace("\x00", " ")
    dates = parse_dates(t)
    pub = dates[0] if dates else None

    scad = None
    for pat in DATE_PATTERNS:
        for m in pat.finditer(t):
            start = max(0, m.start() - 30)
            chunk = t[start:m.end()+30].lower()
            if "scaden" in chunk or "entro" in chunk:
                ds = parse_dates(m.group(0))
                if ds:
                    scad = ds[0]
                    break
        if scad:
            break

    cig = "N/D"
    m = re.search(r"\bCIG\b[:\s\-]*([A-Z0-9]{10})\b", t, flags=re.I)
    if m:
        cig = m.group(1).upper()
    else:
        mm = CIG_RE.search(t)
        if mm:
            cig = mm.group(1).upper()

    cup = "N/D"
    m = re.search(r"\bCUP\b[:\s\-]*([A-Z0-9]{10,15})\b", t, flags=re.I)
    if m:
        cup = m.group(1).upper()
    else:
        mm = CUP_RE.search(t)
        if mm:
            cup = mm.group(1).upper()

    val_tot = parse_first_euro(t)
    val_tot_s = to_euro_str(val_tot) if val_tot is not None else "N/D"

    nome = "N/D"
    for line in t.splitlines()[:40]:
        line2 = line.strip()
        if 8 <= len(line2) <= 180 and not any(x in line2.lower() for x in ["privacy", "cookie", "home", "menu"]):
            nome = line2
            break

    regione = state.get("region") if state.get("region") not in (None, "", "(tutte)") else "N/D"
    stato_progetto = state.get("status") if state.get("status") not in (None, "", "(tutti)") else "N/D"
    fase = state.get("phase") if state.get("phase") not in (None, "", "(tutte)") else "N/D"
    tipologia = state.get("category") if state.get("category") not in (None, "", "(tutte)") else "N/D"
    fonte = state.get("source_label") or host_label(url)

    dettagli = [
        f"Fonte: {fonte}",
        f"URL: {url}",
        f"Data usata filtro: {ddmmyyyy(pub)}",
    ]
    if cig != "N/D":
        dettagli.append(f"CIG: {cig}")
    if cup != "N/D":
        dettagli.append(f"CUP: {cup}")
    if val_tot_s != "N/D":
        dettagli.append(f"Valore stimato: {val_tot_s}")

    return ProjectRecord(
        pubblicazione=ddmmyyyy(pub),
        regione=regione,
        comune="N/D",
        nome_progetto=nome,
        tipologia=tipologia,
        committente="N/D",
        tipo_ente="N/D",
        valore_totale=val_tot_s,
        valore_fase="N/D",
        percento="N/D",
        scadenza=ddmmyyyy(scad),
        cig=cig,
        cup=cup,
        fonte=fonte,
        progettista="N/D",
        fase_progettazione="N/D",
        stato_progetto=stato_progetto,
        fase=fase,
        settore="N/D",
        dettagli="; ".join(dettagli),
    )

def guess_source_from_url(url: str) -> str:
    u = url.lower()
    if "opencup" in u:
        return "OpenCUP"
    if "openpnrr" in u:
        return "OpenPNRR"
    if "acquistinrete" in u or "mepa" in u:
        return "MEPA"
    if "gazzettaufficiale" in u or "guri" in u:
        return "GURI"
    if "anac" in u:
        return "ANAC"
    if "albo" in u and "pretor" in u:
        return "Albo Pretorio"
    return host_label(url)

def extract_links(html: str, base_url: str, limit: int = 200) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        full = normalize_url(urljoin(base_url, href))
        if full:
            links.append(full)
        if len(links) >= limit:
            break
    seen = set()
    out = []
    for x in links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def fetch_text_for_url(url: str, use_selenium: bool, parse_pdf: bool) -> Tuple[str, str]:
    resp = safe_get(url)
    if resp is None:
        if use_selenium:
            html = selenium_fetch(url)
            if html:
                soup = BeautifulSoup(html, "lxml")
                return soup.get_text("\n"), "text/html"
        return "", "error"

    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/pdf" in ctype or url.lower().endswith(".pdf"):
        if parse_pdf:
            return pypdf_extract_text(resp.content), "application/pdf"
        return "", "application/pdf"

    try:
        resp.encoding = resp.apparent_encoding or resp.encoding
    except Exception:
        pass
    soup = BeautifulSoup(resp.text or "", "lxml")
    return soup.get_text("\n"), "text/html"

def within_date_range(pub_str: str, start: Optional[dt.date], end: Optional[dt.date]) -> bool:
    if pub_str in ("", "N/D", None):
        return True
    try:
        d = dt.datetime.strptime(pub_str, "%d/%m/%Y").date()
    except Exception:
        return True
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True

def deduplicate(records: List[ProjectRecord]) -> List[ProjectRecord]:
    def score(r: ProjectRecord) -> int:
        return sum(1 for v in r.to_row().values() if str(v).strip() not in ("", "N/D"))

    by_key: Dict[str, ProjectRecord] = {}
    for r in records:
        if r.cup != "N/D":
            key = "CUP:" + r.cup
        elif r.cig != "N/D":
            key = "CIG:" + r.cig
        else:
            key = "H:" + sha1((r.nome_progetto + "|" + r.fonte).lower())
        if key not in by_key or score(r) > score(by_key[key]):
            by_key[key] = r
    return list(by_key.values())

# -----------------------------
# Export helpers
# -----------------------------
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Progetti")
    bio.seek(0)
    return bio.read()

def build_summary(df: pd.DataFrame) -> str:
    lines = [f"Totale progetti: {len(df)}"]
    vc = df["Fase (Programmazione/Progettazione/Esecuzione/N/D)"].value_counts(dropna=False).head(10)
    lines.append("Distribuzione per Fase: " + ", ".join([f"{k}={v}" for k, v in vc.items()]))
    vc = df["Stato del progetto (PFTE/Definitivo/Esecutivo/N/D)"].value_counts(dropna=False).head(10)
    lines.append("Distribuzione per Stato: " + ", ".join([f"{k}={v}" for k, v in vc.items()]))
    vc = df["Portale/Fonte (OpenPNRR, OpenCUP, MEPA, GURI, Albo Pretorio, etc.)"].value_counts(dropna=False).head(8)
    lines.append("Fonti principali: " + ", ".join([f"{k}={v}" for k, v in vc.items()]))
    return "\n".join(lines)

# -----------------------------
# Pages
# -----------------------------
def page_dataset(state: Dict[str, Any]) -> None:
    st.header("📦 Dataset")
    st.caption("Carica un CSV/XLSX con URL e (opzionale) Regione. Il dataset NON viene usato automaticamente: lo attivi in Ricerca.")

    up = st.file_uploader("Carica dataset (CSV o XLSX)", type=["csv", "xlsx"])
    if up is None:
        st.info("Nessun dataset caricato.")
        return

    try:
        if up.name.lower().endswith(".csv"):
            df = pd.read_csv(up, sep=None, engine="python", on_bad_lines="skip")
        else:
            df = pd.read_excel(up)
    except Exception as e:
        st.error(f"Errore caricamento dataset: {e}")
        return

    if df.empty:
        st.info("Dataset vuoto.")
        return

    st.session_state["dataset_df"] = df
    rcol = detect_region_column(df)
    if rcol:
        st.session_state["dataset_region_col"] = rcol

    st.success(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    st.dataframe(df.head(300), use_container_width=True)

    url_cols = detect_url_columns(df)
    st.markdown("### Colonna URL (opzionale)")
    if url_cols:
        col = st.selectbox("Seleziona colonna URL", url_cols)
        st.session_state["dataset_url_col"] = col
    else:
        st.caption("Non ho trovato colonne con 'url' o 'link' nel nome.")

    st.markdown("### Colonna Regione (opzionale)")
    if rcol:
        st.caption(f"Rilevata colonna Regione: **{rcol}**")
    else:
        st.caption("Non ho rilevato colonna Regione (rinomina includendo 'regione' se vuoi filtro reale).")

def page_search(state: Dict[str, Any]) -> None:
    st.header("🧭 Ricerca")

    st.markdown("### Filtri")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        state["region"] = st.selectbox("Regione", REGIONI, index=REGIONI.index(state.get("region","(tutte)")) if state.get("region") in REGIONI else 0)
    with c2:
        state["category"] = st.selectbox("Tipologia progetto", CATEGORIE, index=CATEGORIE.index(state.get("category","(tutte)")) if state.get("category") in CATEGORIE else 0)
    with c3:
        state["status"] = st.selectbox("Stato progetto", STATI_PROGETTO, index=STATI_PROGETTO.index(state.get("status","(tutti)")) if state.get("status") in STATI_PROGETTO else 0)
    with c4:
        state["phase"] = st.selectbox("Fase", FASE, index=FASE.index(state.get("phase","(tutte)")) if state.get("phase") in FASE else 0)

    st.markdown("### Filtro date (su data del documento quando disponibile)")
    d1, d2 = st.columns([1, 1])
    with d1:
        state["start_date"] = st.date_input("Data inizio (opzionale)", value=None)
    with d2:
        state["end_date"] = st.date_input("Data fine (opzionale)", value=None)

    st.markdown("### Moduli (attivabili)")
    m1, m2, m3 = st.columns([1, 1, 1])
    with m1:
        state["use_selenium"] = st.toggle("Modalità Selenium (lenta)", value=bool(state.get("use_selenium", False)))
    with m2:
        state["parse_pdf"] = st.toggle("Parsing PDF (pypdf)", value=bool(state.get("parse_pdf", True)))
    with m3:
        state["crawl_links"] = st.toggle("Crawl link interni (1 livello)", value=bool(state.get("crawl_links", True)))

    st.markdown("### Sorgenti URL")

    single_url = st.text_input(
        "URL singolo (sito/pagina specifica)",
        value="",
        placeholder="https://www.comune.esempio.it/albo-pretorio",
        help="Inserisci un singolo URL. Verrà aggiunto alla lista finale."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        urls_text = st.text_area(
            "Oppure incolla più URL (uno per riga)",
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

    urls: List[str] = []
    if single_url.strip():
        u = normalize_url(single_url)
        if u:
            urls.append(u)

    if urls_text.strip():
        for line in urls_text.splitlines():
            u = normalize_url(line)
            if u:
                urls.append(u)

    region_selected = state["region"]
    if use_dataset and "dataset_df" in st.session_state:
        df = st.session_state["dataset_df"].copy()
        url_col = st.session_state.get("dataset_url_col")

        region_col = st.session_state.get("dataset_region_col")
        if region_selected != "(tutte)" and region_col and region_col in df.columns:
            df[region_col] = df[region_col].astype(str)
            df = df[df[region_col].str.lower().str.contains(region_selected.lower(), na=False)]

        if url_col and url_col in df.columns:
            from_ds = [normalize_url(x) for x in df[url_col].dropna().astype(str).tolist()]
            from_ds = [x for x in from_ds if x]
            seen = set(urls)
            for u in from_ds:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
            urls = urls[: int(limit_urls)]
        else:
            st.warning("Dataset attivo ma colonna URL non selezionata in Dataset.")

    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]

    if not urls:
        st.info("Inserisci almeno 1 URL (o carica un dataset e seleziona la colonna URL).")
        return

    terms = compile_terms(state["category"], extra_terms, state["status"], state["phase"], state["region"])
    max_pages = st.number_input("Max pagine/URL (solo crawl)", 1, 500, 60, 10)
    state["max_pages"] = int(max_pages)

    if st.button("🚀 Avvia ricerca", type="primary"):
        run_search(urls, terms, state)

def page_results(state: Dict[str, Any]) -> None:
    st.header("📤 Risultati")
    df = st.session_state.get("results_df")
    if df is None or df.empty:
        st.info("Nessun risultato ancora. Vai su 'Ricerca' e avvia una ricerca.")
        return

    st.success(f"Risultati: {len(df)} progetti (deduplicati).")
    st.dataframe(df, use_container_width=True, height=420)

    xlsx = to_excel_bytes(df)
    st.download_button(
        "⬇️ Scarica Excel (.xlsx)",
        data=xlsx,
        file_name="progetti.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    summary = build_summary(df)
    pdf_bytes = reportlab_pdf(df, summary)
    if pdf_bytes:
        st.download_button(
            "⬇️ Scarica PDF (report)",
            data=pdf_bytes,
            file_name="progetti_report.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("PDF non disponibile (manca reportlab). Se vuoi il PDF, aggiungi `reportlab` a requirements.txt.")

    st.markdown("### Riepilogo")
    st.code(summary)

# -----------------------------
# Search runner with progress
# -----------------------------
def run_search(urls: List[str], terms: List[str], state: Dict[str, Any]) -> None:
    use_selenium = bool(state.get("use_selenium", False))
    parse_pdf = bool(state.get("parse_pdf", True))
    crawl_links = bool(state.get("crawl_links", True))
    max_pages = int(state.get("max_pages", 60))

    start_date: Optional[dt.date] = state.get("start_date")
    end_date: Optional[dt.date] = state.get("end_date")

    status_box = st.status("In esecuzione…", expanded=True)
    prog = st.progress(0)
    eta_placeholder = st.empty()

    started = time.time()
    per_item_times: List[float] = []

    visited = set()
    records: List[ProjectRecord] = []

    expanded: List[str] = []
    status_box.write(f"URL base: {len(urls)}")
    for base in urls:
        if base not in expanded:
            expanded.append(base)
        if crawl_links:
            r = safe_get(base)
            if r is not None:
                ctype = (r.headers.get("content-type") or "").lower()
                if "text/html" in ctype:
                    base_host = urlparse(base).netloc.lower()
                    links = extract_links(r.text or "", base, limit=max_pages)
                    for l in links:
                        if urlparse(l).netloc.lower() == base_host and l not in expanded:
                            expanded.append(l)

    target_urls = expanded
    total = len(target_urls)
    done = 0

    def update():
        elapsed = time.time() - started
        avg = (sum(per_item_times) / len(per_item_times)) if per_item_times else None
        if avg:
            remaining = max(0, (total - done) * avg)
            eta_placeholder.info(f"⏱️ Trascorso: {int(elapsed)}s • Stima rimanente: ~{int(remaining)}s • {done}/{total}")
        else:
            eta_placeholder.info(f"⏱️ Trascorso: {int(elapsed)}s • {done}/{total}")
        prog.progress(1.0 if total == 0 else min(1.0, done / total))

    status_box.write(f"URL totali da processare: {total} (crawl={'ON' if crawl_links else 'OFF'})")
    update()

    for url in target_urls:
        if url in visited:
            continue
        visited.add(url)

        t0 = time.time()
        state["source_label"] = guess_source_from_url(url)

        text, _ctype = fetch_text_for_url(url, use_selenium=use_selenium, parse_pdf=parse_pdf)
        done += 1
        per_item_times.append(time.time() - t0)

        if text and text_matches_terms(text, terms):
            rec = extract_from_text(text, url, state)
            if within_date_range(rec.pubblicazione, start_date, end_date):
                records.append(rec)

        update()

    status_box.write("Deduplico e preparo export…")
    records = deduplicate(records)

    df = pd.DataFrame([r.to_row() for r in records])
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = "N/D"
    df = df[OUTPUT_COLUMNS]

    st.session_state["results_df"] = df

    elapsed = time.time() - started
    status_box.update(label=f"Completato in {int(elapsed)}s • risultati: {len(df)}", state="complete")
    prog.progress(1.0)
    update()
    st.success("Ricerca completata. Vai su 'Risultati' per scaricare Excel/PDF.")

# -----------------------------
# App shell
# -----------------------------
def init_state() -> Dict[str, Any]:
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "region": "(tutte)",
            "category": "(tutte)",
            "status": "(tutti)",
            "phase": "(tutte)",
            "use_selenium": False,
            "parse_pdf": True,
            "crawl_links": True,
            "start_date": None,
            "end_date": None,
        }
    return st.session_state["state"]

def main() -> None:
    st.set_page_config(page_title="Ricerca Progetti", layout="wide")
    st.title("🔎 Ricerca Progetti (Streamlit)")

    state = init_state()

    with st.sidebar:
        st.header("Menu")
        page = st.radio("Sezione", ["Dataset", "Ricerca", "Risultati"], index=1)
        st.divider()
        st.caption("Suggerimento: Selenium solo se necessario. Parsing PDF può rallentare.")

    if page == "Dataset":
        page_dataset(state)
    elif page == "Ricerca":
        page_search(state)
    else:
        page_results(state)

if __name__ == "__main__":
    main()
