from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class ProjectHit:
    ente: str
    regione: str
    source_page: str
    pdf_url: str
    titolo: str
    data_atto: Optional[date]
    progettista: str
    email: str
    telefono: str
    cup: str
    cig: str
    importo: str
    descrizione: str


def _safe_str(x: object) -> str:
    return "" if x is None else str(x)


def _requests_get(url: str, *, timeout: int = 20) -> requests.Response:
    return requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf,*/*"},
        allow_redirects=True,
    )


def discover_pdf_links(page_url: str, *, max_links: int = 30, timeout: int = 20) -> List[str]:
    """Estrae link a PDF da una pagina (approccio light: requests+bs4)."""
    try:
        r = _requests_get(page_url, timeout=timeout)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    out: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        if ".pdf" not in href.lower():
            continue
        full = requests.compat.urljoin(page_url, href)
        if full not in out:
            out.append(full)
        if len(out) >= max_links:
            break
    return out


def fetch_pdf_text(pdf_url: str, *, timeout: int = 30, max_pages: int = 12) -> str:
    """Scarica un PDF e prova a estrarre testo via pypdf."""
    try:
        r = _requests_get(pdf_url, timeout=timeout)
        r.raise_for_status()
        reader = PdfReader(BytesIO(r.content))
        texts: List[str] = []
        for i, page in enumerate(reader.pages[:max_pages]):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                texts.append(t)
        return "\n".join(texts)
    except Exception:
        return ""


_RE_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_RE_TEL = re.compile(r"(?:\+?39\s*)?(?:\(?0\d{1,3}\)?\s*)?\d{6,10}")
_RE_CUP = re.compile(r"\b[A-Z]\d{2}[A-Z0-9]{10,}\b")
_RE_CIG = re.compile(r"\b[0-9A-F]{10}\b", re.I)


def _first_match(regex: re.Pattern, text: str) -> str:
    m = regex.search(text or "")
    return m.group(0).strip() if m else ""


def parse_date_from_text(text: str) -> Optional[date]:
    """Cerca una data dd/mm/yyyy o dd-mm-yyyy in un testo."""
    if not text:
        return None
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text)
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d)
    except Exception:
        return None


def parse_project_fields(text: str) -> Dict[str, str | Optional[date]]:
    if not text:
        return {
            "titolo": "",
            "data_atto": None,
            "progettista": "",
            "email": "",
            "telefono": "",
            "cup": "",
            "cig": "",
            "importo": "",
            "descrizione": "",
        }

    # titolo: prima riga non vuota abbastanza lunga
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    titolo = ""
    for ln in lines[:25]:
        if len(ln) >= 25:
            titolo = ln
            break

    email = _first_match(_RE_EMAIL, text)
    tel = _first_match(_RE_TEL, text)
    cup = _first_match(_RE_CUP, text)
    cig = _first_match(_RE_CIG, text)
    data_atto = parse_date_from_text(text)

    # progettista: euristica semplice
    progettista = ""
    for pattern in [
        r"progettist[ai]\s*[:\-]\s*(.{3,80})",
        r"affidatari[oa]\s*[:\-]\s*(.{3,80})",
        r"incaricat[oa]\s*[:\-]\s*(.{3,80})",
        r"studio\s*[:\-]\s*(.{3,80})",
    ]:
        m = re.search(pattern, text, flags=re.I)
        if m:
            progettista = re.sub(r"\s+", " ", m.group(1)).strip()
            break

    # importo
    imp = ""
    m = re.search(r"\b(?:importo|corrispettivo|€|euro)\b[^\n]{0,40}?(\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})?)", text, flags=re.I)
    if m:
        imp = m.group(1)

    descr = ""
    # descrizione: prova a prendere qualche riga dopo 'OGGETTO'
    m = re.search(r"\bOGGETTO\b\s*[:\-]?\s*(.{10,200})", text, flags=re.I)
    if m:
        descr = m.group(1).strip()

    return {
        "titolo": titolo,
        "data_atto": data_atto,
        "progettista": progettista,
        "email": email,
        "telefono": tel,
        "cup": cup,
        "cig": cig,
        "importo": imp,
        "descrizione": descr,
    }


def _get_portal_fields(row: pd.Series) -> Tuple[str, str, str]:
    # Compatibilità: molte versioni del CSV usano nomi colonne diversi.
    ente = (
        row.get("ente")
        or row.get("ENTE")
        or row.get("amministrazione")
        or row.get("AMMINISTRAZIONE")
        or row.get("stazione_appaltante")
        or ""
    )
    regione = row.get("regione") or row.get("REGIONE") or ""
    url = (
        row.get("url")
        or row.get("URL")
        or row.get("albo_url")
        or row.get("ALBO_URL")
        or row.get("link")
        or row.get("LINK")
        or ""
    )
    return _safe_str(ente).strip(), _safe_str(regione).strip(), _safe_str(url).strip()


def search_portals_light(
    portals: pd.DataFrame,
    *,
    start: date,
    end: date,
    include_unknown_date: bool = False,
    max_portals: int = 10,
    max_pdfs_per_portal: int = 10,
    timeout_s: int = 20,
) -> pd.DataFrame:
    """Ricerca "light" pensata per Streamlit Cloud.

    - Non usa Selenium
    - Estrae link PDF dalle pagine fornite nel CSV
    - Prova a leggere testo dal PDF e a tirar fuori campi principali
    - Filtra per data se trovata
    """

    hits: List[ProjectHit] = []

    df = portals.copy()
    if len(df) == 0:
        return pd.DataFrame()

    for _, row in df.head(max_portals).iterrows():
        ente, regione, page_url = _get_portal_fields(row)
        if not page_url:
            continue
        pdfs = discover_pdf_links(page_url, max_links=max_pdfs_per_portal, timeout=timeout_s)
        for pdf_url in pdfs:
            text = fetch_pdf_text(pdf_url, timeout=timeout_s)
            fields = parse_project_fields(text)
            data_atto: Optional[date] = fields["data_atto"]  # type: ignore[assignment]

            # filtro date
            if data_atto is None and not include_unknown_date:
                continue
            if data_atto is not None and not (start <= data_atto <= end):
                continue

            hits.append(
                ProjectHit(
                    ente=ente,
                    regione=regione,
                    source_page=page_url,
                    pdf_url=pdf_url,
                    titolo=str(fields["titolo"] or ""),
                    data_atto=data_atto,
                    progettista=str(fields["progettista"] or ""),
                    email=str(fields["email"] or ""),
                    telefono=str(fields["telefono"] or ""),
                    cup=str(fields["cup"] or ""),
                    cig=str(fields["cig"] or ""),
                    importo=str(fields["importo"] or ""),
                    descrizione=str(fields["descrizione"] or ""),
                )
            )

    out = pd.DataFrame(
        [
            {
                "ENTE": h.ente,
                "REGIONE": h.regione,
                "SOURCE_PAGE": h.source_page,
                "PDF_URL": h.pdf_url,
                "TITOLO": h.titolo,
                "DATA_ATTO": h.data_atto.isoformat() if h.data_atto else "",
                "PROGETTISTA": h.progettista,
                "EMAIL": h.email,
                "TELEFONO": h.telefono,
                "CUP": h.cup,
                "CIG": h.cig,
                "IMPORTO": h.importo,
                "DESCRIZIONE": h.descrizione,
            }
            for h in hits
        ]
    )
    return out
