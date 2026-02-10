from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass
class LoadReport:
    path: str
    rows: int
    cols: int
    delimiter: str
    bad_lines_policy: str
    notes: str = ""


def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # fallback più comune per PA italiana
        return ";" if sample.count(";") > sample.count(",") else ","


def load_csv_robusto(
    source: Any,
    *,
    default_sep: Optional[str] = None,
    encoding: str = "utf-8",
    bad_lines: str = "skip",
    dtype: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, LoadReport]:
    """Carica un CSV in modo resiliente (Streamlit Cloud spesso fallisce su righe sporche).

    - supporta path (str) o file-like (UploadedFile)
    - sniff del delimitatore
    - engine='python' + on_bad_lines per evitare crash
    """

    if isinstance(source, str):
        path = source
        with open(path, "rb") as f:
            raw = f.read(4096)
        sample = raw.decode(encoding, errors="replace")
        sep = default_sep or _sniff_delimiter(sample)

        df = pd.read_csv(
            path,
            sep=sep,
            encoding=encoding,
            engine="python",
            on_bad_lines=bad_lines,
            dtype=dtype,
        )
        report = LoadReport(
            path=path,
            rows=len(df),
            cols=len(df.columns),
            delimiter=sep,
            bad_lines_policy=bad_lines,
            notes="",
        )
        return df, report

    # Streamlit UploadedFile
    name = getattr(source, "name", "uploaded.csv")
    raw = source.getvalue() if hasattr(source, "getvalue") else source.read()
    sample = raw[:4096].decode(encoding, errors="replace")
    sep = default_sep or _sniff_delimiter(sample)
    bio = io.BytesIO(raw)
    df = pd.read_csv(
        bio,
        sep=sep,
        encoding=encoding,
        engine="python",
        on_bad_lines=bad_lines,
        dtype=dtype,
    )
    report = LoadReport(
        path=name,
        rows=len(df),
        cols=len(df.columns),
        delimiter=sep,
        bad_lines_policy=bad_lines,
        notes="",
    )
    return df, report


def normalize_designer_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    if not s:
        return ""
    # rimuovi titoli comuni
    s = re.sub(r"\b(arch\.?|ing\.?|geom\.?|dott\.?|prof\.?|studio|st\.?|soc\.?|srl|spa)\b", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
