# osint_core.py - Core enterprise (fix import + robustezza)
from io import BytesIO
import re
from typing import List, Dict, Optional

import requests
from pypdf import PdfReader

from osint_agent_antibot_v3_2 import BrowserPool, SeleniumCrawler


def extract_project_info_from_pdf(pdf_url: str) -> Dict:
    """Estrae info strutturate da PDF delibera."""
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        reader = PdfReader(BytesIO(response.content))

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        progettista = re.search(
            r"(?:progettista|progettazione|affidatari[oa]|arch\.?|ing\.?)[:\s]*([^\n\r]+)",
            text,
            re.I,
        )
        cup = re.search(r"\bCUP\b[:\s]*([A-Z0-9]+)", text)
        cig = re.search(r"\bCIG\b[:\s]*([A-Z0-9]+)", text)
        importo = re.search(r"(?:importo|valore)[:\s]*€?\s*([\d\.,]+)", text, re.I)
        data = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", text)

        return {
            "progettista_raw": progettista.group(1).strip() if progettista else "non trovato",
            "cup": cup.group(1) if cup else "non trovato",
            "cig": cig.group(1) if cig else "non trovato",
            "importo": importo.group(1) if importo else "non trovato",
            "data_delibera": data.group(1) if data else "non trovato",
            "pdf_source": pdf_url,
            "pdf_text_preview": text[:500],
        }
    except Exception as e:
        return {"error": f"PDF non processabile: {type(e).__name__}", "pdf_url": pdf_url}


def run_scraping(
    capoluoghi: List[Dict],
    start_date,
    end_date,
    categorie: Optional[List[str]] = None,
    max_pdf_per_portale: int = 50,
) -> List[Dict]:
    """Scraping completo con parsing PDF.

    Nota: il filtro data (start_date/end_date) richiede di leggere la data dal PDF/testo.
    Qui viene solo passata per coerenza e possibili estensioni.
    """
    browser_pool = BrowserPool(pool_size=3)
    crawler = SeleniumCrawler(browser_pool)

    all_projects: List[Dict] = []
    try:
        for portal in capoluoghi:
            albo_url = portal.get("ALBO_PRETORIO_URL")
            if not albo_url:
                all_projects.append({**portal, "error": "ALBO_PRETORIO_URL mancante"})
                continue

            try:
                pdf_links = crawler.get_pdf_links(albo_url)
            except Exception as e:
                all_projects.append(
                    {
                        "comune": portal.get("COMUNE"),
                        "provincia": portal.get("PROVINCIA"),
                        "regione": portal.get("REGIONE"),
                        "portal_url": albo_url,
                        "error": f"crawler_error: {type(e).__name__}",
                    }
                )
                continue

            for pdf_url in pdf_links[:max_pdf_per_portale]:
                project_info = extract_project_info_from_pdf(pdf_url)
                project_info.update(
                    {
                        "comune": portal.get("COMUNE"),
                        "provincia": portal.get("PROVINCIA"),
                        "regione": portal.get("REGIONE"),
                        "portal_url": albo_url,
                    }
                )
                all_projects.append(project_info)

    finally:
        browser_pool.cleanup()

    return all_projects
