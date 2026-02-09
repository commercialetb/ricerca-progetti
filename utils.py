# utils.py
from __future__ import annotations

import re
from io import BytesIO
from typing import Dict, List

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def normalize_progettista(nome_raw: str) -> str:
    nome_raw = (nome_raw or "").strip()
    if not nome_raw:
        return "NON TROVATO"

    s = nome_raw.lower()

    # rimuovi titoli comuni
    s = re.sub(r"\b(arch\.?|ing\.?|dott\.?|prof\.?|geom\.?|studio)\b", " ", s, flags=re.I)
    s = re.sub(r"[^a-zàèéìòùA-ZÀÈÉÌÒÙ'\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    parts = [p for p in re.split(r"[ \-]+", s) if p]
    if len(parts) == 1:
        return parts[0].upper()

    # euristica: ultimo token come nome, primo come cognome (meglio di nulla)
    # se già in formato COGNOME NOME, rimane comunque coerente
    cognome = parts[0].upper()
    nome = " ".join(parts[1:]).upper()
    return f"{cognome} {nome}".strip()


def _df_from_projects(projects: List[Dict]) -> pd.DataFrame:
    if not projects:
        return pd.DataFrame()
    df = pd.DataFrame(projects)
    wanted = [
        "progettista_norm",
        "progettista_raw",
        "email",
        "telefono",
        "linkedin",
        "validation_score",
        "lead_quality_score",
        "comune",
        "provincia",
        "regione",
        "data_delibera",
        "importo",
        "cup_cig",
        "pdf_source",
        "portal_url",
    ]
    cols = [c for c in wanted if c in df.columns] + [c for c in df.columns if c not in wanted]
    return df[cols]


def create_excel_4sheets(projects: List[Dict]) -> bytes:
    """Excel 4 sheet: CONTACT_LIST, PROJECTS_BY_DESIGNER, QUALITY_METRICS, OUTREACH_SEGMENTS"""
    df = _df_from_projects(projects)

    wb = Workbook()
    wb.remove(wb.active)

    # CONTACT_LIST
    ws = wb.create_sheet("CONTACT_LIST")
    contact_cols = [c for c in ["progettista_norm", "email", "telefono", "linkedin", "validation_score", "regione", "comune", "pdf_source"] if c in df.columns]
    df_contact = df[contact_cols].copy() if not df.empty else pd.DataFrame(columns=contact_cols)
    for r in dataframe_to_rows(df_contact, index=False, header=True):
        ws.append(r)

    # PROJECTS_BY_DESIGNER
    ws = wb.create_sheet("PROJECTS_BY_DESIGNER")
    if not df.empty and "progettista_norm" in df.columns:
        grouped = df.groupby("progettista_norm").size().reset_index(name="n_progetti").sort_values("n_progetti", ascending=False)
    else:
        grouped = pd.DataFrame(columns=["progettista_norm", "n_progetti"])
    for r in dataframe_to_rows(grouped, index=False, header=True):
        ws.append(r)

    # QUALITY_METRICS
    ws = wb.create_sheet("QUALITY_METRICS")
    qm_cols = [c for c in ["progettista_norm", "validation_score", "lead_quality_score"] if c in df.columns]
    df_qm = df[qm_cols].copy() if not df.empty else pd.DataFrame(columns=qm_cols)
    for r in dataframe_to_rows(df_qm, index=False, header=True):
        ws.append(r)

    # OUTREACH_SEGMENTS
    ws = wb.create_sheet("OUTREACH_SEGMENTS")
    seg = segment_leads(df)
    for r in dataframe_to_rows(seg, index=False, header=True):
        ws.append(r)

    bio = BytesIO()
    wb.save(bio)
    return bio.getvalue()


def segment_leads(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["progettista_norm", "segment", "lead_quality_score", "email", "telefono", "regione", "comune", "pdf_source"])

    out = df.copy()
    if "lead_quality_score" not in out.columns:
        out["lead_quality_score"] = 0.0

    def seg(x: float) -> str:
        try:
            x = float(x)
        except Exception:
            x = 0.0
        if x >= 8.5:
            return "A"
        if x >= 7.0:
            return "B"
        if x >= 5.0:
            return "C"
        return "D"

    out["segment"] = out["lead_quality_score"].apply(seg)

    cols = [c for c in ["progettista_norm", "segment", "lead_quality_score", "email", "telefono", "regione", "comune", "pdf_source"] if c in out.columns]
    return out[cols].sort_values(["segment", "lead_quality_score"], ascending=[True, False])


def create_csv_segments(projects: List[Dict]) -> bytes:
    df = _df_from_projects(projects)
    seg = segment_leads(df)
    return seg.to_csv(index=False).encode("utf-8")


def generate_outreach_templates(top_leads: List[Dict]) -> str:
    lines = []
    lines.append("EMAIL 1\n------")
    lines.append("Oggetto: Riguardo al progetto [PROGETTO] a [CITTÀ]")
    lines.append(
        "Ciao [NOME],\n"
        "ho visto l’intervento [PROGETTO] a [CITTÀ] ([DATA]). "
        "Stiamo lavorando su [SOLUZIONE] che può aiutare in [BENEFIT]. "
        "Ti va una call di 15 minuti questa settimana?\n"
        "Grazie,\n[IL_TUO_NOME]\n"
    )
    lines.append("\nEMAIL 2\n------")
    lines.append("Oggetto: Progetti in ambito [CATEGORIA] in [REGIONE]")
    lines.append(
        "Ciao [NOME],\n"
        "seguo diversi interventi in ambito [CATEGORIA] in [REGIONE]. "
        "Ho notato che hai seguito [N] progetti recenti. "
        "Possiamo sentirci per capire se [SOLUZIONE] è utile anche per i tuoi prossimi lavori?\n"
        "Un saluto,\n[IL_TUO_NOME]\n"
    )

    if top_leads:
        lines.append("\nESEMPI PERSONALIZZATI (TOP)\n--------------------------")
        for p in top_leads[:10]:
            lines.append(
                f"- {p.get('progettista_norm','NON TROVATO')} | {p.get('comune','')} | "
                f"{p.get('data_delibera','')} | fonte: {p.get('pdf_source','')}"
            )

    return "\n".join(lines)
