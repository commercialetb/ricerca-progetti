import csv
import pandas as pd

def load_csv_robusto(path: str, expected_cols: int | None = None):
    """
    Carica un CSV in modo robusto.
    - Prova lettura standard
    - Se fallisce, usa engine=python + handler per righe malformate
    Ritorna: (df, bad_lines)
    """
    bad_lines = []

    # Prima prova: veloce
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if expected_cols and df.shape[1] != expected_cols:
            # se il numero colonne non è quello atteso, forza la modalità robusta
            raise ValueError(f"Colonne lette {df.shape[1]} != attese {expected_cols}")
        return df, bad_lines
    except Exception:
        pass

    # Seconda prova: robusta
    def _bad_line_handler(bad_line):
        bad_lines.append(bad_line)
        return None  # skip

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",
        on_bad_lines=_bad_line_handler,
        encoding_errors="replace",
    )

    # Se vuoi anche “forzare” un numero colonne atteso (opzionale):
    if expected_cols and df.shape[1] > expected_cols:
        df = df.iloc[:, :expected_cols]

    return df, bad_lines
