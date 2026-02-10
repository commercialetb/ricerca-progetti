import io
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from pypdf import PdfReader

def safe_to_datetime(x) -> Optional[pd.Timestamp]:
    if x is None or x == "":
        return None
    try:
        return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return None

def load_master_sa(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        p2 = Path.cwd() / path
        if p2.exists():
            p = p2
        else:
            raise FileNotFoundError(f"File non trovato: {path} (cwd={Path.cwd()})")
    return pd.read_csv(p)

def filter_master_sa(df: pd.DataFrame, regioni, enti, region_col=None, ente_col=None) -> pd.DataFrame:
    out = df.copy()
    if region_col and regioni:
        out = out[out[region_col].isin(regioni)]
    if ente_col and enti:
        out = out[out[ente_col].isin(enti)]
    return out

def extract_pdf_text_basic(pdf_url: str, timeout_s: int = 12, max_pages: int = 5) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RicercaProgetti/1.0)"}
    r = requests.get(pdf_url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    data = io.BytesIO(r.content)
    reader = PdfReader(data)
    txt = []
    for page in reader.pages[:max_pages]:
        try:
            txt.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(txt)

def guess_date_in_text(text: str) -> str:
    m = re.search(r"\b([0-3]\d)[/\.\-]([01]\d)[/\.\-]((?:19|20)\d{2})\b", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    m2 = re.search(r"\b((?:19|20)\d{2})-([01]\d)-([0-3]\d)\b", text)
    if m2:
        return f"{m2.group(3)}/{m2.group(2)}/{m2.group(1)}"
    return ""

def df_to_excel_bytes(results_df: pd.DataFrame, segments_df: pd.DataFrame, meta: Dict[str, Any]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="RESULTS", index=False)
        segments_df.to_excel(writer, sheet_name="SEGMENTS", index=False)
        pd.DataFrame([meta]).to_excel(writer, sheet_name="META", index=False)
    return output.getvalue()

def segment_leads_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def score_row(r):
        s = 0
        if str(r.get("email", "")).strip():
            s += 4
        if str(r.get("phone", "")).strip():
            s += 2
        if str(r.get("cup", "")).strip():
            s += 2
        if str(r.get("cig", "")).strip():
            s += 2
        return s

    out["lead_score_0_10"] = out.apply(score_row, axis=1)

    def seg(x):
        if x >= 8:
            return "A"
        if x >= 6:
            return "B"
        if x >= 4:
            return "C"
        return "D"

    out["segment"] = out["lead_score_0_10"].apply(seg)
    cols = ["region", "ente", "portal_url", "page_url", "pdf_url", "published_date", "email", "phone", "lead_score_0_10", "segment"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values(["segment", "lead_score_0_10"], ascending=[True, False])
