# ai_enrichment.py - optional AI enrichment (safe on Streamlit Cloud)
from __future__ import annotations

import json
from typing import Dict, List

from utils import normalize_progettista

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def ai_enrich_contacts(projects: List[Dict], api_key: str, provider: str) -> List[Dict]:
    """Enrich contacts.

    - If provider SDK missing OR api_key missing: returns projects with placeholders + error_ai.
    """
    enriched: List[Dict] = []
    provider = (provider or "").lower()

    if not api_key:
        for p in projects:
            p = dict(p)
            p.update(
                {
                    "progettista_norm": normalize_progettista(p.get("progettista_raw", "")),
                    "email": "NON TROVATO",
                    "telefono": "NON TROVATO",
                    "linkedin": "NON TROVATO",
                    "validation_score": 0,
                    "fonti": [],
                    "lead_quality_score": 0.0,
                    "error_ai": "API key mancante",
                }
            )
            enriched.append(p)
        return enriched

    if "claude" in provider or "anthropic" in provider:
        try:
            import anthropic  # type: ignore
        except Exception as e:
            for p in projects:
                p = dict(p)
                p.update(
                    {
                        "progettista_norm": normalize_progettista(p.get("progettista_raw", "")),
                        "email": "NON TROVATO",
                        "telefono": "NON TROVATO",
                        "linkedin": "NON TROVATO",
                        "validation_score": 0,
                        "fonti": [],
                        "lead_quality_score": 0.0,
                        "error_ai": f"anthropic non disponibile: {e}",
                    }
                )
                enriched.append(p)
            return enriched

        client = anthropic.Anthropic(api_key=api_key)
        for project in projects:
            p = dict(project)
            prompt = f"""Sei un assistente OSINT.
Trova contatti pubblici e verificabili (se non trovati: 'NON TROVATO') per il progettista:
- progettista_raw: {p.get('progettista_raw')}
- comune: {p.get('comune')}
- provincia: {p.get('provincia')}
- regione: {p.get('regione')}
- fonte pdf: {p.get('pdf_source')}

RISPONDI SOLO con JSON valido con queste chiavi:
{{
  "progettista_norm": "COGNOME NOME",
  "email": "email@dominio.it oppure NON TROVATO",
  "telefono": "+39... oppure NON TROVATO",
  "linkedin": "url oppure NON TROVATO",
  "validation_score": 0-100,
  "fonti": ["..."],
  "lead_quality_score": 1-10
}}
"""
            try:
                resp = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=700,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text
                data = json.loads(text)
                data["progettista_norm"] = normalize_progettista(
                    data.get("progettista_norm") or p.get("progettista_raw", "")
                )
                data["validation_score"] = int(_safe_float(data.get("validation_score", 0), 0))
                data["lead_quality_score"] = _safe_float(data.get("lead_quality_score", 0.0), 0.0)
                p.update(data)
            except Exception as e:
                p.update(
                    {
                        "progettista_norm": normalize_progettista(p.get("progettista_raw", "")),
                        "email": "NON TROVATO",
                        "telefono": "NON TROVATO",
                        "linkedin": "NON TROVATO",
                        "validation_score": 0,
                        "fonti": [],
                        "lead_quality_score": 0.0,
                        "error_ai": f"Errore AI/parsing: {e}",
                    }
                )
            enriched.append(p)
        return enriched

    if "groq" in provider:
        try:
            from groq import Groq  # type: ignore
        except Exception as e:
            for p in projects:
                p = dict(p)
                p.update(
                    {
                        "progettista_norm": normalize_progettista(p.get("progettista_raw", "")),
                        "email": "NON TROVATO",
                        "telefono": "NON TROVATO",
                        "linkedin": "NON TROVATO",
                        "validation_score": 0,
                        "fonti": [],
                        "lead_quality_score": 0.0,
                        "error_ai": f"groq non disponibile: {e}",
                    }
                )
                enriched.append(p)
            return enriched

        client = Groq(api_key=api_key)
        for project in projects:
            p = dict(project)
            prompt = (
                "Estrai contatti pubblici verificabili per progettista "
                f"'{p.get('progettista_raw')}'. "
                "Rispondi SOLO JSON con chiavi: progettista_norm,email,telefono,linkedin,validation_score,fonti,lead_quality_score. "
                "Se non trovato: NON TROVATO."
            )
            try:
                chat = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                text = chat.choices[0].message.content
                data = json.loads(text)
                data["progettista_norm"] = normalize_progettista(
                    data.get("progettista_norm") or p.get("progettista_raw", "")
                )
                data["validation_score"] = int(_safe_float(data.get("validation_score", 0), 0))
                data["lead_quality_score"] = _safe_float(data.get("lead_quality_score", 0.0), 0.0)
                p.update(data)
            except Exception as e:
                p.update(
                    {
                        "progettista_norm": normalize_progettista(p.get("progettista_raw", "")),
                        "email": "NON TROVATO",
                        "telefono": "NON TROVATO",
                        "linkedin": "NON TROVATO",
                        "validation_score": 0,
                        "fonti": [],
                        "lead_quality_score": 0.0,
                        "error_ai": f"Errore Groq/parsing: {e}",
                    }
                )
            enriched.append(p)
        return enriched

    for p in projects:
        p = dict(p)
        p.update(
            {
                "progettista_norm": normalize_progettista(p.get("progettista_raw", "")),
                "email": "NON TROVATO",
                "telefono": "NON TROVATO",
                "linkedin": "NON TROVATO",
                "validation_score": 0,
                "fonti": [],
                "lead_quality_score": 0.0,
                "error_ai": "Provider non supportato",
            }
        )
        enriched.append(p)
    return enriched
