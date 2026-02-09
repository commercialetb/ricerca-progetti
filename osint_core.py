# osint_core.py - core scraping/parsing
from __future__ import annotations

import re
from io import BytesIO
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from osint_agent_antibot_v3_2 import CATEGORIES, SELENIUM_AVAILABLE


def _request_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        }
    )
    return s


def get_pdf_links_requests(url: str, max_pages: int = 5) -> List[str]:
    """Fallback senza Selenium: scarica HTML e cerca link a PDF."""
    found: List[str] = []
    sess = _request_session()

    try:
        r = sess.get(url, timeout=25)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    for a in soup.select("a[href]"):
        href = (a.get("href", "") or "").strip()
        if href and ".pdf" in href.lower():
            full = requests.compat.urljoin(url, href)
            found.append(full)

    # dedup mantenendo ordine
    seen = set()
    out = []
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_project_info_from_pdf(pdf_url: str, portal_url: str = "") -> Dict:
    """Estrae testo dal PDF e tenta di trovare campi chiave.
    OCR è opzionale e viene usato solo se il PDF non ha testo.
    """
    sess = _request_session()
    try:
        r = sess.get(pdf_url, timeout=40)
        r.raise_for_status()
        pdf_bytes = r.content
    except Exception as e:
        return {"pdf_source": pdf_url, "portal_url": portal_url, "error": f"download_failed: {e}"}

    text = ""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        parts: List[str] = []
        for p in reader.pages[:15]:
            t = p.extract_text() or ""
            if t:
                parts.append(t)
        text = "\n".join(parts)
    except Exception:
        text = ""

    # OCR solo se necessario e disponibile
    if not text.strip():
        try:
            from pdf2image import convert_from_bytes  # type: ignore
            import pytesseract  # type: ignore

            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=2)
            ocr_txt = []
            for img in images:
                ocr_txt.append(pytesseract.image_to_string(img, lang="ita+eng"))
            text = "\n".join(ocr_txt)
        except Exception:
            text = ""

    info: Dict[str, object] = {
        "pdf_source": pdf_url,
        "portal_url": portal_url,
        "raw_text_excerpt": (text[:2000] if text else ""),
    }

    if not text:
        info["note"] = "testo_pdf_non_trovato"
        return info

    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I)
    tel_match = re.search(r"(?:\+?39\s*)?(?:0\d{1,3}|3\d{2})[\s\-\./]*\d{5,8}", text)
    cup_match = re.search(r"\bCUP\b\s*[:\-]?\s*([A-Z0-9]{8,20})", text, re.I)
    cig_match = re.search(r"\bCIG\b\s*[:\-]?\s*([A-Z0-9]{6,20})", text, re.I)

    info["email"] = email_match.group(0) if email_match else "non trovato"
    info["telefono"] = tel_match.group(0) if tel_match else "non trovato"
    info["cup_cig"] = " ".join(
        [x for x in [(f"CUP {cup_match.group(1)}" if cup_match else ""), (f"CIG {cig_match.group(1)}" if cig_match else "")] if x]
    ).strip() or "non trovato"

    date_match = re.search(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b", text)
    info["data_delibera"] = date_match.group(1) if date_match else "non trovato"

    imp_match = re.search(r"\b(€|EUR)\s*([0-9\.,]{4,})", text, re.I)
    info["importo"] = (imp_match.group(0) if imp_match else "non trovato")

    progettista = "non trovato"
    for line in text.splitlines():
        if re.search(r"progett|incaric|profession", line, re.I):
            candidate = re.sub(r"\s+", " ", line).strip()
            if 10 <= len(candidate) <= 140:
                progettista = candidate
                break
    info["progettista_raw"] = progettista

    return info


def run_scraping(portals_df, categories: List[str] | None = None, max_pages: int = 3) -> List[Dict]:
    """Scraping multi-portale (Selenium se disponibile, altrimenti fallback requests)."""
    categories = categories or list(CATEGORIES.keys())
    results: List[Dict] = []

    crawler = None
    if SELENIUM_AVAILABLE:
        try:
            from osint_agent_antibot_v3_2 import BrowserPool, SeleniumCrawler

            pool = BrowserPool(pool_size=1, headless=True)
            crawler = SeleniumCrawler(pool, use_human_behavior=False)
        except Exception:
            crawler = None

    for _, row in portals_df.iterrows():
        portal_url = str(row.get("ALBO_PRETORIO_URL", "") or row.get("PORTAL_URL", "")).strip()
        if not portal_url:
            continue

        pdf_links: List[str] = []
        if crawler is not None:
            try:
                pdf_links = list(crawler.get_pdf_links(portal_url, max_pages=max_pages))
            except Exception:
                pdf_links = []

        if not pdf_links:
            pdf_links = get_pdf_links_requests(portal_url, max_pages=max_pages)

        for pdf_url in pdf_links[:30]:
            info = extract_project_info_from_pdf(pdf_url, portal_url=portal_url)
            info["comune"] = row.get("COMUNE", row.get("ENTE", ""))
            info["provincia"] = row.get("PROVINCIA", "")
            info["regione"] = row.get("REGIONE", "")
            results.append(info)

    try:
        if crawler is not None and hasattr(crawler, "browser_pool"):
            crawler.browser_pool.cleanup()
    except Exception:
        pass

    return results
