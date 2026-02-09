# ai_enrichment.py - 7 livelli ricerca contatti + scoring
import anthropic
import json
from typing import List, Dict

def ai_enrich_contacts(projects: List[Dict], api_key: str, provider: str) -> List[Dict]:
    """Arricchisce ogni progetto con contatti verificati"""
    client = anthropic.Anthropic(api_key=api_key)
    
    enriched = []
    for project in projects:
        prompt = f"""
        Per il progettista "{project['progettista_raw']}" del progetto a {project['comune']}:

        RICERCA CONTATTI (7 livelli priorità):
        1. Delibere pubbliche (email/tel diretti)
        2. Ordini professionali
        3. ANAC anticorruzione.it
        4. Sito studio professionale
        5. LinkedIn
        6. Google "nome + email"
        7. Ricerca inversa

        Output JSON valido:
        {{
            "progettista_norm": "COGNOME NOME",
            "email": "email@dominio.it",
            "telefono": "+39...",
            "linkedin": "url",
            "validation_score": 85,
            "fonti": ["Ordine Architetti", "LinkedIn"],
            "lead_quality_score": 8.7
        }}
        """
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            contact_data = json.loads(response.content[0].text)
            project.update(contact_data)
            enriched.append(project)
        except:
            project["error_ai"] = "Parsing JSON fallito"
            enriched.append(project)
    
    return enriched
