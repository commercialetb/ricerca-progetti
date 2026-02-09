# utils.py - helpers (safe, no heavy imports)
from __future__ import annotations

import re
import io
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

TITOLI_PATTERN = re.compile(r"\b(arch\.?|ing\.?|dott\.?|prof\.?|studio)\b\s*", re.IGNORECASE)

def normalize_progettista(nome_raw: str) -> str:
    """Normalizza: 'Arch. Rossi Mario' -> 'ROSSI MARIO' (best-effort)."""
    if not nome_raw:
        return "NON TROVATO"
    s = TITOLI_PATTERN.sub("", str(nome_raw)).strip()
    if not s:
        return "NON TROVATO"
    parts = re.split(r"\s+", s.upper())
    if len(parts) >= 2:
        return f"{parts[-2]} {parts[-1]}"
    return parts[0]

def lead_segment(lead_score: float) -> str:
    if lead_score >= 8.5:
        return "A"
    if lead_score >= 7.0:
        return "B"
    if lead_score >= 5.0:
        return "C"
    return "D"

def create_csv_segments(leads: List[Dict]) -> Tuple[str, Dict[str, str]]:
    """Return combined CSV and per-segment CSVs as strings."""
    df = pd.DataFrame(leads).copy()
    if df.empty:
        return "", {"A": "", "B": "", "C": "", "D": ""}

    if "lead_quality_score" in df.columns and "LEAD_SCORE" not in df.columns:
        df["LEAD_SCORE"] = pd.to_numeric(df["lead_quality_score"], errors="coerce")
    if "LEAD_SCORE" not in df.columns:
        df["LEAD_SCORE"] = 0.0

    df["SEGMENT"] = df["LEAD_SCORE"].fillna(0).apply(lead_segment)

    combined = df.to_csv(index=False)
    per = {}
    for seg in ["A", "B", "C", "D"]:
        per[seg] = df[df["SEGMENT"] == seg].to_csv(index=False)
    return combined, per

def create_excel_4sheets(leads: List[Dict]) -> bytes:
    """Excel 4-sheet: CONTACT_LIST, PROJECTS_BY_DESIGNER, QUALITY_METRICS, OUTREACH_SEGMENTS"""
    wb = Workbook()
    df = pd.DataFrame(leads)

    # 1) CONTACT_LIST
    ws = wb.active
    ws.title = "CONTACT_LIST"
    cols = [c for c in ["progettista_norm", "email", "telefono", "validation_score", "linkedin"] if c in df.columns]
    if not cols:
        cols = list(df.columns)[:10]
    for r in dataframe_to_rows(df[cols], index=False, header=True):
        ws.append(r)

    # 2) PROJECTS_BY_DESIGNER
    ws2 = wb.create_sheet("PROJECTS_BY_DESIGNER")
    proj_cols = [c for c in ["progettista_norm", "comune", "provincia", "regione", "cup_cig", "importo", "data_delibera", "pdf_source", "portal_url"] if c in df.columns]
    if not proj_cols:
        proj_cols = list(df.columns)[:10]
    for r in dataframe_to_rows(df[proj_cols], index=False, header=True):
        ws2.append(r)

    # 3) QUALITY_METRICS
    ws3 = wb.create_sheet("QUALITY_METRICS")
    qm = df.copy()
    if "validation_score" not in qm.columns:
        qm["validation_score"] = None
    if "lead_quality_score" in qm.columns and "LEAD_SCORE" not in qm.columns:
        qm["LEAD_SCORE"] = qm["lead_quality_score"]
    if "LEAD_SCORE" not in qm.columns:
        qm["LEAD_SCORE"] = None
    metric_cols = [c for c in ["progettista_norm", "validation_score", "LEAD_SCORE", "fonti", "error_ai"] if c in qm.columns]
    for r in dataframe_to_rows(qm[metric_cols], index=False, header=True):
        ws3.append(r)

    # 4) OUTREACH_SEGMENTS
    ws4 = wb.create_sheet("OUTREACH_SEGMENTS")
    _combined, per = create_csv_segments(leads)
    seg_df = pd.DataFrame(
        [
            {"SEGMENT": k, "COUNT": len(pd.read_csv(io.StringIO(v))) if v.strip() else 0}
            for k, v in per.items()
        ]
    )
    for r in dataframe_to_rows(seg_df, index=False, header=True):
        ws4.append(r)

    out = BytesIO()
    wb.save(out)
    return out.getvalue()
