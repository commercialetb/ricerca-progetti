# ai_enrichment.py - enrichment contatti (robusto + supporto Groq)
from __future__ import annotations

import json
from typing import List, Dict, Any

from utils import normalize_progettista


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _basic_enrich(project: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback senza chiamate API: normalizzazione + score a 0."""
    project["progettista_norm"] = normalize_progettista(project.get("progettista_raw", "") or "")
    project.setdefault("email", "non trovato")
    project.setdefault("telefono", "non trovato")
    project.setdefault("linkedin", "")
    project.setdefault("validation_score", 0)
    project.setdefault("lead_quality_score", 0.0)
    project.setdefault("fonti", [])
    project.setdefault("error_ai", "API key assente o provider non configurato")
    return project


def ai_enrich_contacts(projects: List[Dict], api_key: str, provider: str) -> List[Dict]:
    """Arricchisce ogni progetto con contatti verificati.
    Se api_key è vuota, usa fallback basic (no crash).
    """
    provider = (provider or "").strip()

    if not api_key:
        return [_basic_enrich(dict(p)) for p in projects]

    enriched: List[Dict] = []

    if provider == "Claude (Anthropic)":
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)

        def call_llm(prompt: str) -> str:
            resp = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

    elif provider == "Groq":
        from groq import Groq  # type: ignore
        client = Groq(api_key=api_key)

        def call_llm(prompt: str) -> str:
            resp = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=900,
            )
            return resp.choices[0].message.content or ""

    else:
        return [_basic_enrich(dict(p)) for p in projects]

    for project in projects:
        p = dict(project)

        prompt = f"""Per il progettista "{p.get('progettista_raw', 'non trovato')}" del progetto a {p.get('comune', '')}:

RICERCA CONTATTI (7 livelli priorità):
1. Delibere pubbliche (email/tel diretti)
2. Ordini professionali
3. ANAC anticorruzione.it
4. Sito studio professionale
5. LinkedIn
6. Google "nome + email"
7. Ricerca inversa

IMPORTANTISSIMO:
- Restituisci SOLO JSON valido (senza markdown).
- Se un dato non è trovato usa "non trovato".

JSON atteso:
{{
  "progettista_norm": "COGNOME NOME",
  "email": "non trovato",
  "telefono": "non trovato",
  "linkedin": "",
  "validation_score": 0,
  "fonti": [],
  "lead_quality_score": 0.0
}}
"""

        try:
            raw = call_llm(prompt).strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start : end + 1]

            contact_data = json.loads(raw)

            p.update(contact_data)

            if not p.get("progettista_norm") or p["progettista_norm"] == "non trovato":
                p["progettista_norm"] = normalize_progettista(p.get("progettista_raw", ""))

            p["validation_score"] = int(_safe_float(p.get("validation_score", 0), 0))
            p["lead_quality_score"] = _safe_float(p.get("lead_quality_score", 0.0), 0.0)

            enriched.append(p)
        except Exception as e:
            p = _basic_enrich(p)
            p["error_ai"] = f"AI error: {type(e).__name__}"
            enriched.append(p)

    return enriched
