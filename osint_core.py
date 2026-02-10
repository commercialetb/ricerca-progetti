import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import urllib.parse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils import safe_to_datetime, extract_pdf_text_basic, guess_date_in_text

ProgressCB = Optional[Callable[[int, int, str], None]]
LogCB = Optional[Callable[[str], None]]

@dataclass
class FoundItem:
    region: str
    ente: str
    portal_url: str
    page_url: str
    title: str
    doc_type: str
    published_date: str
    keywords_hit: str
    pdf_url: str
    cup: str
    cig: str
    amount: str
    designer: str
    email: str
    phone: str
    extraction_notes: str

def default_keywords_by_category() -> Dict[str, List[str]]:
    return {
        "Sport": ["palazzetto", "piscina", "palestra", "stadio", "polisportivo", "impianto sportivo", "campo", "tennis", "nuoto", "arena"],
        "Piste ciclabili": ["pista ciclabile", "ciclopedonale", "ciclovia", "greenway", "bike lane", "ciclo", "mobilità dolce", "percorso cicl", "anello ciclabile", "ciclo-pedonale"],
        "Riqualificazione urbana": ["riqualificazione", "rigenerazione", "piano urbano", "restyling", "recupero", "rifunzionalizzazione", "arredo urbano", "manutenzione straordinaria", "centro storico", "piazza"],
        "Sanitario": ["ospedale", "poliambulatorio", "RSA", "pronto soccorso", "terapia intensiva", "dialisi", "consultorio", "ambulatorio", "struttura sanitaria", "casa della comunità"],
        "Musei": ["museo", "pinacoteca", "galleria", "allestimento", "musealizzazione", "centro culturale", "spazio espositivo", "restauro", "museo civico", "museo d'arte"],
        "Università": ["campus", "aula", "laboratorio", "dipartimento", "studentato", "biblioteca", "polo universitario", "mensa universitaria", "residenza studenti", "università"],
        "Innovazione": ["hub", "incubatore", "coworking", "fab lab", "innovazione", "digital", "smart", "laboratorio", "centro ricerca", "competence center"],
        "ERP": ["edilizia residenziale pubblica", "ERP", "case popolari", "alloggi", "housing", "residenze", "edilizia pubblica", "ristrutturazione alloggi", "riqualificazione energetica", "manutenzione alloggi"],
        "TPL": ["trasporto pubblico", "autostazione", "capolinea", "deposito bus", "tram", "metropolitana", "fermata", "stazione", "intermodalità", "bus"],
        "Edilizia giudiziaria": ["tribunale", "procura", "palazzo di giustizia", "uffici giudiziari", "corte", "giudice", "aula udienza", "carcere", "casa circondariale", "ufficio giudice pace"],
        "Archivi": ["archivio", "deposito archivistico", "biblioteca", "archivio storico", "digitalizzazione archivi", "polo archivistico", "riordino archivi", "archivistica", "sala consultazione", "conservazione"],
        "Intrattenimento": ["teatro", "cinema", "auditorium", "arena", "spettacolo", "sala eventi", "centro congressi", "palco", "festival", "anfiteatro"],
        "Aeroporti": ["aeroporto", "terminal", "pista", "hangar", "aerostazione", "piazzale", "sicurezza aeroportuale", "parcheggi aeroporto"],
        "Centri commerciali": ["centro commerciale", "galleria commerciale", "retail park", "ipermercato", "mall", "parcheggio", "ampliamento", "negozi", "food court"],
        "Alberghi": ["hotel", "albergo", "resort", "ospitalità", "camere", "spa", "struttura ricettiva", "boutique hotel"],
    }

def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if not u.startswith("http"):
        u = "https://" + u
    return u

def _fetch(url: str, timeout_s: int) -> Optional[str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RicercaProgetti/1.0; +https://streamlit.app)"}
    r = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=True)
    if r.status_code >= 400:
        return None
    return r.text

def _extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        links.append(urllib.parse.urljoin(base_url, href))

    seen = set()
    out = []
    for x in links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _keyword_hit(text: str, keywords: List[str]) -> List[str]:
    t = (text or "").lower()
    hits = []
    for k in keywords:
        kk = k.lower().strip()
        if kk and kk in t:
            hits.append(k)
    return hits

def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.text.strip() if soup.title and soup.title.text else ""
    return title[:200]

def run_osint_search(
    portals_df: pd.DataFrame,
    url_col: str,
    region_col: Optional[str],
    ente_col: Optional[str],
    date_start: str,
    date_end: str,
    categories: List[str],
    custom_keywords: List[str],
    search_mode: str,
    include_pdf_parse: bool,
    max_pages: int,
    request_timeout: int,
    features: Dict,
    progress_cb: ProgressCB = None,
    log_cb: LogCB = None,
) -> Tuple[pd.DataFrame, Dict]:
    kw_map = default_keywords_by_category()
    keywords: List[str] = []
    for c in categories:
        keywords.extend(kw_map.get(c, []))
    keywords.extend(custom_keywords or [])

    # de-dup
    seen = set()
    keywords = [k for k in keywords if k and not (k.lower() in seen or seen.add(k.lower()))]

    ds = safe_to_datetime(date_start)
    de = safe_to_datetime(date_end)

    found: List[FoundItem] = []
    total = len(portals_df)

    for i, row in enumerate(portals_df.to_dict(orient="records"), start=1):
        portal_url = _normalize_url(str(row.get(url_col, "")))
        region = str(row.get(region_col, "")) if region_col else ""
        ente = str(row.get(ente_col, "")) if ente_col else ""

        msg = f"({i}/{total}) Scansione: {ente or 'ENTE?'} — {portal_url}"
        if progress_cb:
            progress_cb(i - 1, total, msg)
        if log_cb:
            log_cb(msg)

        try:
            html = _fetch(portal_url, timeout_s=request_timeout)
        except Exception as e:
            if log_cb:
                log_cb(f"  ✗ errore fetch: {type(e).__name__}: {e}")
            continue

        if not html:
            if log_cb:
                log_cb("  ✗ nessuna risposta / status >=400")
            continue

        root_title = _extract_title(html)
        root_hits = _keyword_hit(html + " " + root_title, keywords)
        root_links = _extract_links(portal_url, html)

        if search_mode == "Solo elenco portali (test connessione)":
            found.append(FoundItem(
                region=region, ente=ente, portal_url=portal_url, page_url=portal_url,
                title=root_title or "(home)", doc_type="PORTAL_OK",
                published_date="", keywords_hit=",".join(root_hits[:10]),
                pdf_url="", cup="", cig="", amount="", designer="", email="", phone="",
                extraction_notes="Connectivity OK",
            ))
            continue

        # pick only a few relevant pages to avoid loops
        priority_terms = ["albo", "trasparente", "delib", "determin", "affid", "atti", "bandi", "gara", "provved"]
        cand = []
        for u in root_links:
            lu = u.lower()
            score = sum(1 for t in priority_terms if t in lu)
            if score > 0:
                cand.append((score, u))
        cand.sort(reverse=True, key=lambda x: x[0])

        pages = [portal_url] + [u for _, u in cand[:max_pages - 1]]
        pages = pages[:max_pages]

        for p in pages:
            try:
                ph = _fetch(p, timeout_s=request_timeout)
            except Exception as e:
                if log_cb:
                    log_cb(f"  - skip page fetch error: {p} ({type(e).__name__})")
                continue
            if not ph:
                continue

            page_title = _extract_title(ph) or p
            hits = _keyword_hit(ph + " " + page_title, keywords)
            if not hits and p != portal_url:
                continue

            links = _extract_links(p, ph)
            pdfs = [u for u in links if u.lower().endswith(".pdf") or ".pdf?" in u.lower()]

            for pdf_url in pdfs[:30]:  # hard cap per page
                published = ""
                cup = cig = ""
                email = phone = ""
                notes = ""

                if include_pdf_parse:
                    try:
                        text = extract_pdf_text_basic(pdf_url, timeout_s=request_timeout)
                        published = guess_date_in_text(text) or ""

                        cup_m = re.search(r"\bCUP\b[:\s]*([A-Z0-9]{6,})", text, flags=re.I)
                        cig_m = re.search(r"\bCIG\b[:\s]*([A-Z0-9]{6,})", text, flags=re.I)
                        cup = cup_m.group(1).strip() if cup_m else ""
                        cig = cig_m.group(1).strip() if cig_m else ""

                        email_m = re.search(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", text, flags=re.I)
                        email = email_m.group(1) if email_m else ""

                        phone_m = re.search(r"\b(?:\+?39)?\s*(?:0\d{1,3}|\d{3})[\s\-./]?\d{5,8}\b", text)
                        phone = phone_m.group(0).strip() if phone_m else ""

                        if published:
                            dt = safe_to_datetime(published)
                            if dt is not None and ds is not None and de is not None and (dt < ds or dt > de):
                                continue
                    except Exception as e:
                        notes = f"PDF parse error: {type(e).__name__}"
                else:
                    notes = "PDF parse disabilitato"

                found.append(FoundItem(
                    region=region, ente=ente, portal_url=portal_url, page_url=p,
                    title=page_title[:200], doc_type="PDF_LINK",
                    published_date=published,
                    keywords_hit=",".join(hits[:12]),
                    pdf_url=pdf_url, cup=cup, cig=cig,
                    amount="", designer="", email=email, phone=phone,
                    extraction_notes=notes,
                ))

        if progress_cb:
            progress_cb(i, total, f"Completato {i}/{total}")

    df = pd.DataFrame([f.__dict__ for f in found])
    meta = {
        "portals_scanned": int(total),
        "date_start": date_start,
        "date_end": date_end,
        "keywords_count": len(keywords),
        "search_mode": search_mode,
        "include_pdf_parse": include_pdf_parse,
    }
    return df, meta
