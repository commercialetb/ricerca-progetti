import time
import pandas as pd
import streamlit as st

from osint_core import run_osint_search, default_keywords_by_category
from utils import (
    load_master_sa,
    filter_master_sa,
    df_to_excel_bytes,
    segment_leads_placeholder,
)

APP_TITLE = "Ricerca Progetti Italia — Streamlit (Anti-loop)"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ----------------------------
# Session state defaults
# ----------------------------
if "master_df" not in st.session_state:
    st.session_state.master_df = None
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "last_run_meta" not in st.session_state:
    st.session_state.last_run_meta = {}
if "config" not in st.session_state:
    st.session_state.config = None

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log_lines.append(f"[{ts}] {msg}")

# ----------------------------
# Sidebar: menu
# ----------------------------
st.sidebar.title("Menu")
section = st.sidebar.radio(
    "Sezione",
    ["1) Dati portali", "2) Configura ricerca", "3) Esegui", "4) Risultati & Export", "Impostazioni"],
)

st.title(APP_TITLE)
st.caption("Boot veloce: niente scraping / OCR / Selenium all'avvio. Tutto parte solo quando premi **Avvia ricerca**.")

# ----------------------------
# Frontend toggles (feature flags)
# ----------------------------
with st.sidebar.expander("Feature (toggle)", expanded=False):
    enable_plotly = st.toggle("Abilita grafici Plotly (se installato)", value=False)
    enable_ai = st.toggle("Abilita AI enrichment (se installato/configurato)", value=False)
    enable_selenium = st.toggle("Abilita Selenium (richiede chromium + driver)", value=False)
    enable_ocr = st.toggle("Abilita OCR PDF (richiede poppler + tesseract)", value=False)

st.session_state["features"] = {
    "plotly": enable_plotly,
    "ai": enable_ai,
    "selenium": enable_selenium,
    "ocr": enable_ocr,
}

# ----------------------------
# 1) Load portals (MASTER_SA)
# ----------------------------
if section == "1) Dati portali":
    st.subheader("1) Carica portali (MASTER_SA)")
    st.write("Puoi usare il CSV nel repo (consigliato) oppure caricarne uno manualmente.")

    col1, col2 = st.columns([1, 1])

    with col1:
        default_path = st.text_input(
            "Percorso CSV MASTER_SA nel repo",
            value="MASTER_SA_gare_links_NORMALIZED.csv",
            help="Metti il file nella root del repo. In alternativa usa upload qui sotto."
        )
        if st.button("Carica dal repo", type="primary"):
            try:
                df = load_master_sa(default_path)
                st.session_state.master_df = df
                log(f"MASTER_SA caricato dal repo: {default_path} ({len(df)} righe)")
                st.success(f"Caricato: {len(df)} righe")
            except Exception as e:
                st.error(f"Errore caricamento dal repo: {e}")
                log(f"ERRORE load_master_sa: {e}")

    with col2:
        up = st.file_uploader("Oppure carica CSV MASTER_SA", type=["csv"])
        if up is not None and st.button("Usa CSV caricato"):
            try:
                df = pd.read_csv(up)
                st.session_state.master_df = df
                log(f"MASTER_SA caricato da upload ({len(df)} righe)")
                st.success(f"Caricato: {len(df)} righe")
            except Exception as e:
                st.error(f"Errore lettura CSV: {e}")
                log(f"ERRORE read_csv(upload): {e}")

    if st.session_state.master_df is not None:
        st.divider()
        st.subheader("Anteprima e filtri rapidi")

        df = st.session_state.master_df.copy()
        cols = {c.lower(): c for c in df.columns}
        region_col = cols.get("regione") or cols.get("region")
        ente_col = cols.get("ente") or cols.get("amministrazione") or cols.get("stazione_appaltante")
        url_col = cols.get("url") or cols.get("link") or cols.get("sito") or cols.get("domain")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            regioni = sorted(df[region_col].dropna().unique().tolist()) if region_col else []
            sel_reg = st.multiselect("Regioni", regioni, default=[])
        with c2:
            enti = sorted(df[ente_col].dropna().unique().tolist()) if ente_col else []
            sel_ente = st.multiselect("Enti", enti, default=[])
        with c3:
            st.write("Colonne rilevate")
            st.json({"regione": region_col, "ente": ente_col, "url": url_col})

        df_f = filter_master_sa(df, sel_reg, sel_ente, region_col=region_col, ente_col=ente_col)
        st.write(f"Righe dopo filtro: **{len(df_f)}**")
        st.dataframe(df_f.head(200), use_container_width=True)

# ----------------------------
# 2) Configure search
# ----------------------------
elif section == "2) Configura ricerca":
    st.subheader("2) Configura ricerca")

    master = st.session_state.master_df
    if master is None:
        st.warning("Prima carica il MASTER_SA in **1) Dati portali**.")
        st.stop()

    cols = {c.lower(): c for c in master.columns}
    region_col = cols.get("regione") or cols.get("region")
    ente_col = cols.get("ente") or cols.get("amministrazione") or cols.get("stazione_appaltante")
    url_col = cols.get("url") or cols.get("link") or cols.get("sito") or cols.get("domain")

    st.markdown("### Target (filtri portali)")
    c1, c2 = st.columns([1, 1])
    with c1:
        regioni = sorted(master[region_col].dropna().unique().tolist()) if region_col else []
        sel_reg = st.multiselect("Regioni (opzionale)", regioni, default=[])
    with c2:
        enti = sorted(master[ente_col].dropna().unique().tolist()) if ente_col else []
        sel_ente = st.multiselect("Enti (opzionale)", enti, default=[])

    st.markdown("### Periodo atti (obbligatorio)")
    c3, c4 = st.columns([1, 1])
    with c3:
        date_start = st.date_input("Data inizio", value=pd.to_datetime("2025-07-01").date())
    with c4:
        date_end = st.date_input("Data fine", value=pd.to_datetime("2026-01-28").date())

    st.markdown("### Modalità ricerca")
    search_mode = st.selectbox(
        "Tipo di ricerca",
        [
            "Delibere/Affidamenti (HTML + PDF link)",
            "Solo PDF (filetype:pdf link già in pagina)",
            "Solo elenco portali (test connessione)"
        ],
        index=0
    )

    st.markdown("### Categorie + keyword")
    categories = list(default_keywords_by_category().keys())
    sel_cats = st.multiselect("Categorie", categories, default=["Riqualificazione urbana", "Sport"])
    custom_kw = st.text_area(
        "Keyword aggiuntive (una per riga, opzionale)",
        value="affidamento progettazione\nincarico professionale\nprogetto esecutivo\nCUP\nCIG\n",
        height=120
    )

    st.markdown("### Limiti anti-blocco (consigliato)")
    c5, c6, c7 = st.columns([1, 1, 1])
    with c5:
        max_portals = st.number_input("Max portali per run", min_value=1, max_value=5000, value=30, step=1)
    with c6:
        max_pages = st.number_input("Max pagine per portale", min_value=1, max_value=50, value=5, step=1)
    with c7:
        request_timeout = st.number_input("Timeout request (sec)", min_value=3, max_value=60, value=12, step=1)

    st.markdown("### Output")
    include_pdf_parse = st.checkbox("Apri e parse PDF (pypdf) quando trovati", value=True)

    st.session_state.config = {
        "sel_reg": sel_reg,
        "sel_ente": sel_ente,
        "date_start": str(date_start),
        "date_end": str(date_end),
        "search_mode": search_mode,
        "sel_cats": sel_cats,
        "custom_kw": [k.strip() for k in custom_kw.splitlines() if k.strip()],
        "max_portals": int(max_portals),
        "max_pages": int(max_pages),
        "request_timeout": int(request_timeout),
        "include_pdf_parse": bool(include_pdf_parse),
        "columns": {"region_col": region_col, "ente_col": ente_col, "url_col": url_col},
    }

    st.success("Configurazione salvata. Vai su **3) Esegui**.")

# ----------------------------
# 3) Run
# ----------------------------
elif section == "3) Esegui":
    st.subheader("3) Esegui ricerca (on-demand)")

    if st.session_state.master_df is None:
        st.warning("Prima carica il MASTER_SA in **1) Dati portali**.")
        st.stop()
    if st.session_state.config is None:
        st.warning("Prima configura la ricerca in **2) Configura ricerca**.")
        st.stop()

    cfg = st.session_state.config
    st.json(cfg, expanded=False)

    if st.button("Avvia ricerca", type="primary"):
        st.session_state.results_df = pd.DataFrame()
        st.session_state.log_lines = []
        log("Avvio ricerca...")

        master = st.session_state.master_df.copy()
        cols = cfg["columns"]
        region_col, ente_col, url_col = cols["region_col"], cols["ente_col"], cols["url_col"]

        master_f = filter_master_sa(master, cfg["sel_reg"], cfg["sel_ente"], region_col=region_col, ente_col=ente_col)
        if url_col is None:
            st.error("Non trovo la colonna URL nel MASTER_SA. Rinominare una colonna in 'url' oppure 'link'.")
            log("ERRORE: colonna url non trovata.")
            st.stop()

        master_f = master_f.dropna(subset=[url_col]).head(cfg["max_portals"])
        log(f"Portali selezionati: {len(master_f)} (max {cfg['max_portals']})")

        with st.spinner("Ricerca in corso..."):
            progress = st.progress(0)
            status = st.empty()

            def cb(i, total, msg):
                progress.progress(int((i / max(total, 1)) * 100))
                status.write(msg)

            results_df, meta = run_osint_search(
                portals_df=master_f,
                url_col=url_col,
                region_col=region_col,
                ente_col=ente_col,
                date_start=cfg["date_start"],
                date_end=cfg["date_end"],
                categories=cfg["sel_cats"],
                custom_keywords=cfg["custom_kw"],
                search_mode=cfg["search_mode"],
                include_pdf_parse=cfg["include_pdf_parse"],
                max_pages=cfg["max_pages"],
                request_timeout=cfg["request_timeout"],
                features=st.session_state.get("features", {}),
                progress_cb=cb,
                log_cb=log,
            )

        st.session_state.results_df = results_df
        st.session_state.last_run_meta = meta
        log("Ricerca completata.")
        st.success(f"Completato: {len(results_df)} record trovati.")

    st.divider()
    st.subheader("Log (ultimo run)")
    st.code("\n".join(st.session_state.log_lines[-200:]) or "Nessun log (ancora).")

# ----------------------------
# 4) Results & Export
# ----------------------------
elif section == "4) Risultati & Export":
    st.subheader("4) Risultati & Export")

    df = st.session_state.results_df
    if df is None or df.empty:
        st.info("Nessun risultato. Vai su **3) Esegui** e lancia una ricerca.")
        st.stop()

    meta = st.session_state.get("last_run_meta", {})
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Record", len(df))
    with c2:
        st.metric("Portali scansionati", meta.get("portals_scanned", 0))
    with c3:
        st.write(f"Periodo: **{meta.get('date_start','?')} → {meta.get('date_end','?')}**")

    st.dataframe(df, use_container_width=True, height=520)

    st.markdown("### Segmentazione (placeholder)")
    seg_df = segment_leads_placeholder(df)
    st.dataframe(seg_df, use_container_width=True)

    st.markdown("### Export")
    excel_bytes = df_to_excel_bytes(results_df=df, segments_df=seg_df, meta=meta)

    st.download_button(
        "⬇️ Scarica Excel (risultati + segmenti)",
        data=excel_bytes,
        file_name="ricerca_progetti_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "⬇️ Scarica CSV risultati",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="results.csv",
        mime="text/csv",
    )

# ----------------------------
# Settings / Diagnostics
# ----------------------------
else:
    st.subheader("Impostazioni & Diagnostica")
    st.write("Verifica installazione feature opzionali senza rompere l'avvio.")

    checks = {}
    for mod in ["plotly.express", "selenium", "pytesseract", "pdf2image", "anthropic", "groq"]:
        try:
            __import__(mod.split(".")[0])
            checks[mod] = "OK"
        except Exception as e:
            checks[mod] = f"NON INSTALLATO ({type(e).__name__})"
    st.json(checks)

    st.markdown("### Performance (Streamlit Cloud)")
    st.write(
        "- `packages.txt` vuoto = boot rapido.\n"
        "- Abilita OCR/Selenium solo se serve davvero.\n"
        "- Tieni bassi: max portali / max pagine / timeout.\n"
        "- Nessun import pesante in top-level (già sistemato)."
    )
