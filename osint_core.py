# osint_core.py - lightweight core suitable for Streamlit Cloud
from __future__ import annotations

import re
from io import BytesIO
from typing import Dict, List
from urllib.parse import urljoin, urldefrag

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from utils import normalize_progettista

PDF_EXT_RE = re.compile(r"\.pdf($|\?)", re.IGNORECASE)

def _abs(base: str, href: str) -> str:
    href = href.strip()
    href, _ = urldefrag(href)
    return urljoin(base, href)

def get_pdf_links_requests(seed_url: str, max_pages: int = 5, timeout: int = 20) -> List[str]:
    """Best-effort PDF link discovery using requests+bs4 (no JS)."""
    seen_pages = set()
    to_visit = [seed_url]
    pdfs: List[str] = []
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; OSINTAgent/1.0)"}

    while to_visit and len(seen_pages) < max_pages:
        url = to_visit.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)

        try:
            r = session.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(r.text, "lxml")

        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if not href:
                continue
            full = _abs(url, href)
            if PDF_EXT_RE.search(full) and full not in pdfs:
                pdfs.append(full)

        # naive pagination discovery
        for a in soup.select("a[href]"):
            txt = (a.get_text(" ", strip=True) or "").lower()
            href = a.get("href", "")
            if not href:
                continue
            if any(k in txt for k in ["next", "successiva", "successivo", "pagina", ">>", "›"]):
                full = _abs(url, href)
                if full not in seen_pages and full not in to_visit:
                    to_visit.append(full)

    return pdfs

def extract_project_info_from_pdf(pdf_url: str, timeout: int = 30) -> Dict:
    """Extract structured info from a PDF (text-only via pypdf)."""
    try:
        r = requests.get(pdf_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        reader = PdfReader(BytesIO(r.content))

        text_parts = []
        for page in reader.pages[:20]:
            text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts)

        progettista = re.search(
            r"(?:progettista|affidatari[oa]|incaricat[oa]|arch\.?|ing\.?)\s*[:\-]?\s*(.+?)(?:\n|$)",
            text,
            re.I,
        )
        cup = re.search(r"\bCUP\b\s*[:\-]?\s*([A-Z0-9]{6,20})", text)
        cig = re.search(r"\bCIG\b\s*[:\-]?\s*([A-Z0-9]{6,20})", text)
        importo = re.search(
            r"(?:importo|valore|corrispettivo)\s*[:\-]?\s*€?\s*([\d\.]+,\d{2}|[\d,\.]+)",
            text,
            re.I,
        )
        data = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", text)

        progettista_raw = progettista.group(1).strip() if progettista else "NON TROVATO"

        return {
            "progettista_raw": progettista_raw,
            "progettista_norm": normalize_progettista(progettista_raw),
            "cup": cup.group(1) if cup else "NON TROVATO",
            "cig": cig.group(1) if cig else "NON TROVATO",
            "cup_cig": (cup.group(1) if cup else (cig.group(1) if cig else "NON TROVATO")),
            "importo": importo.group(1) if importo else "NON TROVATO",
            "data_delibera": data.group(1) if data else "NON TROVATO",
            "pdf_source": pdf_url,
            "pdf_text_preview": text[:600],
        }
    except Exception as e:
        return {"error": f"PDF non processabile: {e}", "pdf_source": pdf_url}

def run_scraping_light(
    capoluoghi: List[Dict], max_pdf_per_portale: int = 20, max_pages_per_portale: int = 5
) -> List[Dict]:
    """Light scraping: find PDF links with requests, then parse PDFs with pypdf."""
    out: List[Dict] = []
    for portal in capoluoghi:
        portal_url = portal.get("ALBO_PRETORIO_URL") or portal.get("PORTAL_URL") or portal.get("url")
        if not portal_url:
            continue
        pdf_links = get_pdf_links_requests(portal_url, max_pages=max_pages_per_portale)
        for pdf_url in pdf_links[:max_pdf_per_portale]:
            info = extract_project_info_from_pdf(pdf_url)
            info.update(
                {
                    "comune": portal.get("COMUNE") or portal.get("comune"),
                    "provincia": portal.get("PROVINCIA") or portal.get("provincia"),
                    "regione": portal.get("REGIONE") or portal.get("regione"),
                    "portal_url": portal_url,
                }
            )
            out.append(info)
    return out

def run_scraping_selenium(capoluoghi: List[Dict], max_pdf_per_portale: int = 30) -> List[Dict]:
    """Heavy scraping: uses SeleniumCrawler if available (optional)."""
    from osint_agent_antibot_v3_2 import BrowserPool, SeleniumCrawler, SELENIUM_AVAILABLE

    if not SELENIUM_AVAILABLE:
        raise RuntimeError(
            "Selenium non disponibile (pip selenium e system chromium/chromedriver richiesti)."
        )

    browser_pool = BrowserPool(pool_size=2)
    crawler = SeleniumCrawler(browser_pool)

    out: List[Dict] = []
    try:
        for portal in capoluoghi:
            portal_url = portal.get("ALBO_PRETORIO_URL")
            if not portal_url:
                continue
            pdf_links = crawler.get_pdf_links(portal_url)
            for pdf_url in pdf_links[:max_pdf_per_portale]:
                info = extract_project_info_from_pdf(pdf_url)
                info.update(
                    {
                        "comune": portal.get("COMUNE"),
                        "provincia": portal.get("PROVINCIA"),
                        "regione": portal.get("REGIONE"),
                        "portal_url": portal_url,
                    }
                )
                out.append(info)
    finally:
        browser_pool.cleanup()

    return out
