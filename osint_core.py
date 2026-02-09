# osint_core.py - Core enterprise
from osint_agent_antibot_v3_2 import BrowserPool, SeleniumCrawler, CATEGORIES
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import re
from typing import List, Dict
import requests

def extract_project_info_from_pdf(pdf_url: str) -> Dict:
    """Estrae info strutturate da PDF delibera"""
    try:
        response = requests.get(pdf_url, timeout=30)
        reader = PdfReader(BytesIO(response.content))
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # Regex per info chiave
        progettista = re.search(r"(?:progettista|progettazione|arch\.?|ing\.?)[:\s]*(.+?)(?:\n|$)", text, re.I)
        cup_cig = re.search(r"(CUP|CIG)[:\s]*([A-Z0-9]+)", text)
        importo = re.search(r"(?:importo|valore)[:\s]*€?([\d\.,]+)", text)
        data = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", text)
        
        return {
            "progettista_raw": progettista.group(1) if progettista else "non trovato",
            "cup_cig": cup_cig.group(2) if cup_cig else "non trovato",
            "importo": importo.group(1) if importo else "non trovato",
            "data_delibera": data.group(1) if data else "non trovato",
            "pdf_source": pdf_url,
            "pdf_text_preview": text[:500]
        }
    except:
        return {"error": "PDF non processabile", "pdf_url": pdf_url}

def run_scraping(capoluoghi: List[Dict], start_date, end_date) -> List[Dict]:
    """Scraping completo con parsing PDF"""
    browser_pool = BrowserPool(pool_size=3)
    crawler = SeleniumCrawler(browser_pool)
    
    all_projects = []
    for portal in capoluoghi:
        pdf_links = crawler.get_pdf_links(portal["ALBO_PRETORIO_URL"])
        
        for pdf_url in pdf_links[:50]:  # max 50 PDF per portale
            project_info = extract_project_info_from_pdf(pdf_url)
            project_info.update({
                "comune": portal.get("COMUNE"),
                "provincia": portal.get("PROVINCIA"),
                "regione": portal.get("REGIONE"),
                "portal_url": portal["ALBO_PRETORIO_URL"]
            })
            all_projects.append(project_info)
    
    browser_pool.cleanup()
    return all_projects
