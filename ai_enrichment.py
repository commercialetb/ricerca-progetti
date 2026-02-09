# ai_enrichment.py - Arricchimento contatti (opzionale)
from __future__ import annotations

import json
from typing import Dict, List, Optional

from utils import normalize_progettista


def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def ai_enrich_contacts(projects: List[Dict], api_key: str | None = None, provider: str = "anthropic") -> List[Dict]:
    """Arricchisce i progetti con contatti.
    - Se manca API key o provider non è installato: NON fallisce, mette 'non trovato'.
    - Non inventa mai email/telefono.
    """
    provider = (provider or "anthropic").lower().strip()
    api_key = (api_key or "").strip()

    if not api_key:
        out: List[Dict] = []
        for p in projects or []:
            q = dict(p)
            q.setdefault("progettista_norm", normalize_progettista(q.get("progettista_raw", "")))
            q.setdefault("email", "non trovato")
            q.setdefault("telefono", "non trovato")
            q.setdefault("linkedin", "non trovato")
            q.setdefault("validation_score", 0)
            q.setdefault("fonti", [])
            q.setdefault("lead_quality_score", 0)
            q["ai_note"] = "API key non impostata: enrichment saltato"
            out.append(q)
        return out

    client = None
    if provider == "anthropic":
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            client = None
            provider_err = f"anthropic non disponibile: {e}"
    elif provider == "groq":
        try:
            from groq import Groq  # type: ignore
            client = Groq(api_key=api_key)
        except Exception as e:
            client = None
            provider_err = f"groq non disponibile: {e}"
    else:
        client = None
        provider_err = f"provider sconosciuto: {provider}"

    if client is None:
        out: List[Dict] = []
        for p in projects or []:
            q = dict(p)
            q.setdefault("progettista_norm", normalize_progettista(q.get("progettista_raw", "")))
            q.setdefault("email", "non trovato")
            q.setdefault("telefono", "non trovato")
            q.setdefault("linkedin", "non trovato")
            q.setdefault("validation_score", 0)
            q.setdefault("fonti", [])
            q.setdefault("lead_quality_score", 0)
            q["ai_note"] = provider_err
            out.append(q)
        return out

    enriched: List[Dict] = []
    for project in projects or []:
        base = dict(project)
        base.setdefault("progettista_norm", normalize_progettista(base.get("progettista_raw", "")))

        prompt = (
            "Sei un assistente OSINT. NON inventare mai email o telefoni. "
            "Se non trovi un dato, rispondi con 'non trovato'.\n\n"
            f"Progettista (raw): {base.get('progettista_raw','')}\n"
            f"Comune: {base.get('comune','')}\n"
            f"Regione: {base.get('regione','')}\n"
            f"Fonte PDF: {base.get('pdf_source','')}\n\n"
            "Trova (se esistono) email, telefono, LinkedIn del progettista, citando fonti. "
            "Rispondi SOLO con JSON valido nel seguente schema:\n"
            "{"
            "\"progettista_norm\":\"COGNOME NOME\","
            "\"email\":\"email@dominio.it o non trovato\","
            "\"telefono\":\"+39... o non trovato\","
            "\"linkedin\":\"url o non trovato\","
            "\"validation_score\":0,"
            "\"fonti\":[\"...\"],"
            "\"lead_quality_score\":0"
            "}"
        )

        raw_text = None
        try:
            if provider == "anthropic":
                resp = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=700,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = resp.content[0].text
            else:
                resp = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                raw_text = resp.choices[0].message.content

            data = json.loads(raw_text)
        except Exception as e:
            base["error_ai"] = f"Parsing/Call fallita: {e}"
            base.setdefault("email", "non trovato")
            base.setdefault("telefono", "non trovato")
            base.setdefault("linkedin", "non trovato")
            base.setdefault("validation_score", 0)
            base.setdefault("fonti", [])
            base.setdefault("lead_quality_score", 0)
            enriched.append(base)
            continue

        base["progettista_norm"] = (data.get("progettista_norm") or base["progettista_norm"]).strip() or base["progettista_norm"]
        base["email"] = (data.get("email") or "non trovato").strip() or "non trovato"
        base["telefono"] = (data.get("telefono") or "non trovato").strip() or "non trovato"
        base["linkedin"] = (data.get("linkedin") or "non trovato").strip() or "non trovato"

        vs = data.get("validation_score")
        try:
            base["validation_score"] = int(float(vs)) if vs is not None else 0
        except Exception:
            base["validation_score"] = 0

        lqs = _safe_float(data.get("lead_quality_score"), 0.0) or 0.0
        base["lead_quality_score"] = max(0.0, min(10.0, float(lqs)))

        fonti = data.get("fonti")
        base["fonti"] = [str(x) for x in fonti][:10] if isinstance(fonti, list) else []

        enriched.append(base)

    return enriched
