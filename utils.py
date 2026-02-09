# utils.py
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def normalize_progettista(nome_raw: str) -> str:
    """Normalizza: "Arch. Rossi Mario" → "ROSSI MARIO" """
    # Rimuovi titoli
    titoli = ["arch.", "ing.", "dott.", "prof.", "studio"]
    for titolo in titoli:
        nome_raw = re.sub(rf"\b{titolo}\.?\s*", "", nome_raw, flags=re.I)
    
    # Maiuscole + COGNOME NOME
    parts = nome_raw.strip().upper().split()
    return " ".join(reversed(parts[:2])) if len(parts) >= 2 else nome_raw.upper()

def create_excel_4sheets(leads: List[Dict]) -> bytes:
    """Excel con 4 sheet come richiesto"""
    wb = Workbook()
    
    # Sheet 1: CONTACT_LIST
    df_contacts = pd.DataFrame(leads)[["progettista_norm", "email", "telefono", "validation_score"]]
    ws1 = wb.active
    ws1.title = "CONTACT_LIST"
    for r in dataframe_to_rows(df_contacts, index=False):
        ws1.append(r)
    
    # Sheet 2: PROJECTS_BY_DESIGNER
    # ... implementazione simile
    
    # Salva in bytes
    from io import BytesIO
    output = BytesIO()
    wb.save(output)
    return output.getvalue()

def generate_outreach_templates(top_leads: List[Dict]) -> str:
    """Genera email personalizzate"""
    templates = []
    for lead in top_leads:
        template = f"""
        Subject: Complimenti per il progetto {lead['comune']}!
        
        Gentile {lead['progettista_norm']},
        
        Ho visto il suo recente progetto a {lead['comune']} (CUP: {lead['cup_cig']}).
        Bellissimo lavoro! Vorrei proporle la nostra soluzione per...
        
        Possiamo fissare una breve call?
        
        Cordiali saluti,
        [Tuo Nome]
        """
        templates.append(template)
    return "\n\n".join(templates)
