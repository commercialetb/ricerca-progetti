import os
from datetime import date

import pandas as pd
import streamlit as st

from osint_core import search_portals_light
from utils import dataframe_to_csv_bytes, load_csv_robusto


APP_TITLE = "🧠 Ricerca Progetti – OSINT Agent (Streamlit)"
DEFAULT_MASTER_CSV = "MASTER_SA_gare_links_NORMALIZED.csv"  # deve essere presente nel repo


def init_page() -> None:
    st.set_page_config(page_title="Ricerca Progetti", layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "App semplificata per evitare loop in fase di boot: i carichi pesanti sono *opt-in* dal menu a sinistra."
    )


def sidebar_settings() -> dict:
    st.sidebar.header("⚙️ Menu")
    page = st.sidebar.radio(
        "Sezione",
        ["1) Dataset", "2) Ricerca", "3) Risultati & Export", "4) Impostazioni"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Modalità")
    enable_pdf_parsing = st.sidebar.toggle("Parsing PDF (pypdf)", value=True)
    include_unknown_date = st.sidebar.toggle("Includi record senza data", value=False)

    # Opzioni "pesanti": le mostriamo, ma non importiamo nulla se non abilitate.
    st.sidebar.subheader("Opzioni avanzate (opt-in)")
    enable_selenium = st.sidebar.toggle("Selenium/Chromium (se installato)", value=False)
    enable_ocr = st.sidebar.toggle("OCR (pdf2image+tesseract) (se installato)", value=False)
    enable_ai = st.sidebar.toggle("AI Enrichment (se configurato)", value=False)

    return {
        "page": page,
        "enable_pdf_parsing": enable_pdf_parsing,
        "include_unknown_date": include_unknown_date,
        "enable_selenium": enable_selenium,
        "enable_ocr": enable_ocr,
        "enable_ai": enable_ai,
    }


def page_dataset() -> None:
    st.subheader("📦 Caricamento dataset portali")
    st.write(
        "Per evitare blocchi all'avvio, il CSV non viene caricato automaticamente: scegli tu quando caricarlo."
    )

    src = st.radio(
        "Fonte dataset",
        ["Dal repo (DEFAULT)", "Upload CSV"],
        horizontal=True,
    )

    default_path = os.path.join(os.getcwd(), DEFAULT_MASTER_CSV)
    if src == "Dal repo (DEFAULT)":
        st.code(default_path)
        if st.button("Carica CSV dal repo", type="primary"):
            df, report = load_csv_robusto(default_path)
            st.session_state["portals_df"] = df
            st.session_state["portals_report"] = report
            st.success(f"Caricati {len(df):,} record dal CSV.")

    else:
        up = st.file_uploader("Carica CSV portali", type=["csv"], accept_multiple_files=False)
        if up is not None:
            df, report = load_csv_robusto(up)
            st.session_state["portals_df"] = df
            st.session_state["portals_report"] = report
            st.success(f"Caricati {len(df):,} record dal CSV caricato.")

    report = st.session_state.get("portals_report")
    if report is not None:
        with st.expander("ℹ️ Report caricamento CSV"):
            st.write(f"**Sorgente:** {report.path}")
            st.write(f"**Righe:** {report.rows:,}")
            st.write(f"**Colonne:** {report.cols}")
            st.write(f"**Delimitatore:** `{report.delimiter}`")
            st.write(f"**Bad lines policy:** `{report.bad_lines_policy}`")
            if getattr(report, "notes", ""):
                st.info(report.notes)

    df = st.session_state.get("portals_df")
    if isinstance(df, pd.DataFrame):
        st.caption("Anteprima dataset")
        st.dataframe(df.head(50), use_container_width=True)


def page_search(opts: dict) -> None:
    st.subheader("🔎 Ricerca")

    df = st.session_state.get("portals_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Prima carica il dataset in **1) Dataset**.")
        return

    # Filtri base
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        start_d = st.date_input("Data inizio", value=date(2025, 7, 1))
    with col2:
        end_d = st.date_input("Data fine", value=date(2026, 1, 28))
    with col3:
        max_portals = st.number_input("Max portali", min_value=1, max_value=5000, value=30, step=1)
    with col4:
        max_pdfs = st.number_input("Max PDF per portale", min_value=1, max_value=200, value=10, step=1)

    # Filtri per regione/comune se presenti
    region_col = "REGIONE" if "REGIONE" in df.columns else None
    comune_col = "COMUNE" if "COMUNE" in df.columns else None

    f1, f2 = st.columns([1, 1])
    with f1:
        regioni = []
        if region_col:
            all_reg = sorted([x for x in df[region_col].dropna().unique().tolist() if str(x).strip()])
            regioni = st.multiselect("Regioni", options=all_reg, default=[])
        else:
            st.caption("Colonna REGIONE non trovata nel CSV.")
    with f2:
        comuni = []
        if comune_col:
            all_com = sorted([x for x in df[comune_col].dropna().unique().tolist() if str(x).strip()])
            comuni = st.multiselect("Comuni", options=all_com, default=[])
        else:
            st.caption("Colonna COMUNE non trovata nel CSV.")

    filtered = df.copy()
    if region_col and regioni:
        filtered = filtered[filtered[region_col].isin(regioni)]
    if comune_col and comuni:
        filtered = filtered[filtered[comune_col].isin(comuni)]

    url_col = "ALBO_PRETORIO_URL" if "ALBO_PRETORIO_URL" in filtered.columns else None
    if not url_col:
        st.error("Nel CSV non trovo la colonna ALBO_PRETORIO_URL.")
        return

    st.caption(f"Portali selezionati: {len(filtered):,}")
    st.dataframe(filtered.head(20), use_container_width=True)

    st.divider()
    st.subheader("Esecuzione")
    colA, colB = st.columns([2, 1])
    with colA:
        st.write(
            "La modalità *Light* usa solo requests+BeautifulSoup (più stabile su Streamlit Cloud). "
            "Le opzioni Selenium/OCR/AI sono solo "
            "togliere/aggiungere in futuro: qui le abiliti dal menu, ma se non sono nel requirements l'app non si rompe."
        )
    with colB:
        run = st.button("🚀 Avvia ricerca", type="primary", use_container_width=True)

    if not run:
        return

    with st.spinner("Ricerca in corso..."):
        results = search_portals_light(
            portals_df=filtered.head(int(max_portals)),
            start_date=start_d,
            end_date=end_d,
            max_pdfs_per_portal=int(max_pdfs),
            parse_pdf=bool(opts["enable_pdf_parsing"]),
            include_unknown_date=bool(opts["include_unknown_date"]),
        )

    st.session_state["results_df"] = pd.DataFrame(results)
    st.success(f"Completato. Record estratti: {len(st.session_state['results_df']):,}")


def page_results() -> None:
    st.subheader("📤 Risultati & Export")
    df = st.session_state.get("results_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Nessun risultato ancora. Vai su **2) Ricerca**.")
        return

    st.dataframe(df, use_container_width=True, height=520)
    st.divider()

    csv_bytes = dataframe_to_csv_bytes(df)
    st.download_button(
        "⬇️ Scarica risultati (CSV)",
        data=csv_bytes,
        file_name="risultati_osint.csv",
        mime="text/csv",
        use_container_width=True,
    )


def page_settings(opts: dict) -> None:
    st.subheader("🧩 Impostazioni / Diagnostica")
    st.write("Questa sezione serve per capire subito *cosa è installato* e cosa no.")

    st.markdown("#### Stato dipendenze opzionali")

    def _check(modname: str) -> str:
        try:
            __import__(modname)
            return "✅"
        except Exception:
            return "❌"

    rows = [
        ("plotly", _check("plotly")),
        ("selenium", _check("selenium")),
        ("pdf2image", _check("pdf2image")),
        ("pytesseract", _check("pytesseract")),
        ("anthropic", _check("anthropic")),
        ("groq", _check("groq")),
    ]
    st.table(pd.DataFrame(rows, columns=["Modulo", "Disponibile"]))

    st.markdown(
        "**Nota:** se attivi Selenium/OCR/AI dal menu ma qui vedi ❌, devi aggiungere le dipendenze in requirements.txt."
    )

    st.markdown("#### Opzioni attuali")
    st.json(opts)


def main() -> None:
    init_page()
    opts = sidebar_settings()

    page = opts["page"]
    if page.startswith("1"):
        page_dataset()
    elif page.startswith("2"):
        page_search(opts)
    elif page.startswith("3"):
        page_results()
    else:
        page_settings(opts)


if __name__ == "__main__":
    main()
