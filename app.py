import streamlit as st

st.set_page_config(page_title="Ricerca Progetti", layout="wide")

st.title("Ricerca Progetti — OSINT Agent")

# ---------------------------
# Helpers: import opzionali
# ---------------------------
def optional_import(module_name: str):
    try:
        module = __import__(module_name)
        return module, None
    except Exception as e:
        return None, str(e)

def check_cmd_exists(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None

# ---------------------------
# Sidebar: Feature flags
# ---------------------------
st.sidebar.header("⚙️ Modalità Avanzate (opzionali)")

enable_ocr = st.sidebar.toggle("Abilita OCR (PDF scannerizzati)", value=False)
enable_selenium = st.sidebar.toggle("Abilita Selenium (anti-bot / siti dinamici)", value=False)

st.sidebar.caption(
    "Nota: l’attivazione richiede che le dipendenze siano installate nel deploy "
    "(requirements.txt / packages.txt). Se mancano, l’app non va in errore: ti avvisa."
)

# ---------------------------
# Validazione dipendenze
# ---------------------------
ocr_ready = True
selenium_ready = True

if enable_ocr:
    # python deps
    pytesseract, err1 = optional_import("pytesseract")
    pdf2image, err2 = optional_import("pdf2image")
    PIL, err3 = optional_import("PIL")

    # system deps
    has_tesseract = check_cmd_exists("tesseract")
    has_pdftoppm = check_cmd_exists("pdftoppm")  # poppler

    missing = []
    if err1: missing.append("pytesseract (pip)")
    if err2: missing.append("pdf2image (pip)")
    if err3: missing.append("Pillow/PIL (pip)")
    if not has_tesseract: missing.append("tesseract (apt)")
    if not has_pdftoppm: missing.append("poppler-utils (apt)")

    if missing:
        ocr_ready = False
        st.warning(
            "OCR richiesto ma non disponibile su questo deploy.\n\n"
            f"**Mancano:** {', '.join(missing)}\n\n"
            "➡️ Per abilitarlo davvero: aggiungi i pacchetti in requirements.txt / packages.txt e ridistribuisci."
        )

if enable_selenium:
    selenium, err = optional_import("selenium")
    has_chromium = check_cmd_exists("chromium") or check_cmd_exists("chromium-browser")
    has_driver = check_cmd_exists("chromedriver")

    missing = []
    if err: missing.append("selenium (pip)")
    if not has_chromium: missing.append("chromium (apt)")
    if not has_driver: missing.append("chromium-driver / chromedriver (apt)")

    if missing:
        selenium_ready = False
        st.warning(
            "Selenium richiesto ma non disponibile su questo deploy.\n\n"
            f"**Mancano:** {', '.join(missing)}\n\n"
            "➡️ Per abilitarlo davvero: aggiungi i pacchetti in requirements.txt / packages.txt e ridistribuisci."
        )

# ---------------------------
# UI principale (placeholder)
# ---------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input")
    url = st.text_input("URL / Portale da analizzare", placeholder="https://...")
    run = st.button("Avvia ricerca", type="primary", use_container_width=True)

with col2:
    st.subheader("Stato moduli")
    st.write(f"🧾 OCR: {'✅ pronto' if (enable_ocr and ocr_ready) else ('🟡 disattivo' if not enable_ocr else '❌ mancante')}")
    st.write(f"🌐 Selenium: {'✅ pronto' if (enable_selenium and selenium_ready) else ('🟡 disattivo' if not enable_selenium else '❌ mancante')}")

# ---------------------------
# Esecuzione: NON avviare Selenium/OCR all'import!
# ---------------------------
if run:
    if not url:
        st.error("Inserisci un URL.")
        st.stop()

    st.info("Esecuzione… (demo)")

    # Esempio routing:
    if enable_selenium and selenium_ready:
        st.success("Userò Selenium per questa ricerca (quando integrato nel tuo core).")
        # qui chiamerai il tuo modulo, MA SOLO dentro al click
        # from osint_agent_antibot_v3_2 import run_with_selenium
        # results = run_with_selenium(url, ...)
    else:
        st.success("Userò modalità standard (requests/bs4).")
        # from osint_core import run_standard
        # results = run_standard(url, ...)

    if enable_ocr and ocr_ready:
        st.success("OCR attivo: userò OCR solo per PDF scannerizzati (quando integrato).")

    st.write("✅ Fine (placeholder). Integra qui le chiamate reali ai tuoi moduli.")
