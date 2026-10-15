from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Gangguan Analisis Ca · SSA-Nyala",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HTML_FILE = Path(__file__).parent / "index.html"

if not HTML_FILE.exists():
    st.error("File index.html tidak ditemukan. Pastikan index.html ikut diunggah ke repository GitHub.")
    st.stop()

html = HTML_FILE.read_text(encoding="utf-8")

# Seluruh antarmuka dan interaksi aplikasi asli ditampilkan dari index.html.
# JavaScript Three.js untuk model 3D memerlukan koneksi internet karena memakai CDN.
components.html(html, height=1100, scrolling=True)
