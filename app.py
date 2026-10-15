import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================ FUNGSI PERHITUNGAN
def linreg(x, y):
    """Regresi linear OLS y = m*x + b beserta statistik kalibrasi."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("Minimal 3 titik kalibrasi.")
    if np.ptp(x) == 0:
        raise ValueError("Konsentrasi standar harus bervariasi.")
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sy_x = np.sqrt(ss_res / (n - 2))  # standar deviasi residual
    sxx = float(np.sum((x - x.mean()) ** 2))
    se_m = sy_x / np.sqrt(sxx)
    se_b = sy_x * np.sqrt(1 / n + x.mean() ** 2 / sxx)
    return dict(m=m, b=b, r2=r2, r=np.sign(m) * np.sqrt(max(r2, 0)),
                sy_x=sy_x, se_m=se_m, se_b=se_b, n=n)


def conc_from_abs(a, m, b):
    """Konsentrasi terbaca (mg/L) dari absorbansi: C = (A - b) / m."""
    return (np.asarray(a, dtype=float) - b) / m


def lod_loq(m, sy_x=None, sd_blank=None):
    """LOD = 3,3*s/m ; LOQ = 10*s/m. Pakai SD blanko bila ada, kalau tidak Sy/x."""
    s = sd_blank if (sd_blank is not None and sd_blank > 0) else sy_x
    return 3.3 * s / m, 10 * s / m


def analyze_interference(df, m, b, fp_default=1.0, alpha=0.05):
    """
    df kolom: Perlakuan, Pengganggu (mg/L), Ca teoritis (mg/L), A1..An, FP
    Baris pertama dengan Pengganggu == 0 (atau baris pertama) dipakai sebagai kontrol.
    """
    a_cols = [c for c in df.columns if c.startswith("A") and c[1:].isdigit()]
    rows, conc_by_row = [], []
    for _, r in df.iterrows():
        a = pd.to_numeric(r[a_cols], errors="coerce").dropna().to_numpy(dtype=float)
        fp = r.get("FP", fp_default)
        fp = fp_default if pd.isna(fp) else float(fp)
        c = conc_from_abs(a, m, b) * fp  # konsentrasi tiap ulangan, sudah x FP
        conc_by_row.append(c)
        rows.append(dict(
            Perlakuan=r["Perlakuan"],
            **{"Pengganggu (mg/L)": r["Pengganggu (mg/L)"]},
            **{"Ca teoritis (mg/L)": r["Ca teoritis (mg/L)"]},
            n=len(a),
            **{"Rata-rata A": a.mean() if len(a) else np.nan},
            **{"SD A": a.std(ddof=1) if len(a) > 1 else np.nan},
            **{"Ca terukur (mg/L)": c.mean() if len(c) else np.nan},
            **{"SD (mg/L)": c.std(ddof=1) if len(c) > 1 else np.nan},
        ))
    out = pd.DataFrame(rows)
    out["%RSD"] = out["SD (mg/L)"] / out["Ca terukur (mg/L)"].abs() * 100
    out["% Recovery"] = out["Ca terukur (mg/L)"] / out["Ca teoritis (mg/L)"] * 100

    ctrl_idx = out.index[out["Pengganggu (mg/L)"] == 0]
    ctrl_idx = ctrl_idx[0] if len(ctrl_idx) else 0
    ctrl_mean = out.loc[ctrl_idx, "Ca terukur (mg/L)"]
    out["% Gangguan"] = (out["Ca terukur (mg/L)"] - ctrl_mean) / ctrl_mean * 100

    pvals = []
    for i, c in enumerate(conc_by_row):
        ctrl = conc_by_row[ctrl_idx]
        if i == ctrl_idx or len(c) < 2 or len(ctrl) < 2:
            pvals.append(np.nan)
        else:
            pvals.append(stats.ttest_ind(c, ctrl, equal_var=False).pvalue)
    out["p (uji t vs kontrol)"] = pvals
    out["Signifikan"] = [("Ya" if (not np.isnan(p) and p < alpha) else
                          ("-" if np.isnan(p) else "Tidak")) for p in pvals]
    return out, ctrl_idx


# ============================================================ TAMPILAN APLIKASI
st.set_page_config(page_title="Analisis Gangguan Ca", page_icon="🧪", layout="wide")
st.title("🧪 Analisis Gangguan Ca")
st.caption("Spektrofotometri Serapan Atom (AAS) nyala · alat 3D")

# ---------------------------------------------------------------- 1. KALIBRASI
st.header("1. Kurva Kalibrasi Ca")
st.write("Masukkan absorbansi standar Ca. Gunakan konsentrasi 0 untuk blanko.")

default_cal = pd.DataFrame({
    "Konsentrasi (mg/L)": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "Absorbansi": [0.002, 0.051, 0.100, 0.148, 0.199, 0.247],  # contoh, ganti dengan data Anda
})
cal = st.data_editor(default_cal, num_rows="dynamic", use_container_width=True, key="cal")
cal = cal.apply(pd.to_numeric, errors="coerce").dropna()

blank_txt = st.text_input(
    "Absorbansi ulangan blanko (opsional, pisahkan dengan koma) — untuk LOD/LOQ",
    placeholder="0.001, 0.002, 0.001, 0.003")
subtract_blank = st.checkbox("Koreksi absorbansi dengan rata-rata blanko", value=False,
                             help="Biasanya tidak perlu bila blanko 0 mg/L sudah ada di kurva.")

try:
    x = cal["Konsentrasi (mg/L)"].to_numpy()
    y = cal["Absorbansi"].to_numpy()
    blank = None
    if blank_txt.strip():
        blank = np.array([float(v) for v in blank_txt.replace(";", ",").split(",") if v.strip()])
    if subtract_blank and blank is not None and len(blank):
        y = y - blank.mean()
    fit = linreg(x, y)
except Exception as e:
    st.error(f"Kalibrasi belum valid: {e}")
    st.stop()

m, b = fit["m"], fit["b"]
sd_blank = blank.std(ddof=1) if (blank is not None and len(blank) > 1) else None
lod, loq = lod_loq(m, sy_x=fit["sy_x"], sd_blank=sd_blank)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Slope (m)", f"{m:.5f}")
c2.metric("Intersep (b)", f"{b:.5f}")
c3.metric("R²", f"{fit['r2']:.5f}")
c4.metric("n titik", fit["n"])
c5, c6 = st.columns(2)
c5.metric("LOD (mg/L)", f"{lod:.3f}")
c6.metric("LOQ (mg/L)", f"{loq:.3f}")
st.latex(rf"A = {m:.5f}\,C \;{'+' if b >= 0 else '-'}\; {abs(b):.5f}")

if fit["r2"] < 0.995:
    st.warning("R² < 0,995: linearitas kurang baik. Periksa standar yang menyimpang atau persempit rentang.")
if m <= 0:
    st.error("Slope ≤ 0: periksa data kalibrasi.")
    st.stop()

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.scatter(x, y, color="C0", label="Standar")
xx = np.linspace(x.min(), x.max(), 100)
ax.plot(xx, m * xx + b, color="C3", label="Regresi")
ax.set_xlabel("Konsentrasi Ca (mg/L)")
ax.set_ylabel("Absorbansi")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# ---------------------------------------------------------------- 2. GANGGUAN
st.header("2. Data Gangguan")
st.write(
    "Baris dengan **Pengganggu = 0** dipakai sebagai kontrol (Ca saja). "
    "Isi absorbansi tiap ulangan (A1–A3). **FP** = faktor pengenceran sampel. "
    "Kosongkan sel ulangan yang tidak dipakai.")

default_int = pd.DataFrame({
    "Perlakuan": ["Kontrol (Ca saja)", "Ca + pengganggu 1", "Ca + pengganggu 2", "Ca + pengganggu 3"],
    "Pengganggu (mg/L)": [0.0, 10.0, 50.0, 100.0],
    "Ca teoritis (mg/L)": [3.0, 3.0, 3.0, 3.0],
    "A1": [0.148, 0.141, 0.128, 0.110],   # contoh, ganti dengan data Anda
    "A2": [0.149, 0.140, 0.127, 0.111],
    "A3": [0.147, 0.142, 0.129, 0.109],
    "FP": [1.0, 1.0, 1.0, 1.0],
})
data = st.data_editor(default_int, num_rows="dynamic", use_container_width=True, key="int")

alpha = st.select_slider("Taraf signifikansi (α)", options=[0.01, 0.05, 0.10], value=0.05)

try:
    res, ctrl_idx = analyze_interference(data, m, b, alpha=alpha)
except Exception as e:
    st.error(f"Data gangguan belum valid: {e}")
    st.stop()

st.subheader("Hasil")
fmt = {
    "Rata-rata A": "{:.4f}", "SD A": "{:.4f}", "Ca terukur (mg/L)": "{:.3f}",
    "SD (mg/L)": "{:.3f}", "%RSD": "{:.2f}", "% Recovery": "{:.1f}",
    "% Gangguan": "{:+.2f}", "p (uji t vs kontrol)": "{:.4f}",
}
st.dataframe(res.style.format(fmt, na_rep="-"), use_container_width=True)

# peringatan rentang kalibrasi
a_cols = [c for c in data.columns if c.startswith("A") and c[1:].isdigit()]
a_all = pd.to_numeric(data[a_cols].stack(), errors="coerce").dropna()
if len(a_all) and (a_all.max() > y.max() * 1.05 or a_all.min() < y.min() - 0.005):
    st.warning("Ada absorbansi di luar rentang kalibrasi — hasil ekstrapolasi kurang andal. "
               "Encerkan sampel dan isi FP.")

# grafik
fig2, ax2 = plt.subplots(figsize=(6, 3.6))
ax2.errorbar(res["Pengganggu (mg/L)"], res["% Recovery"],
             yerr=(res["SD (mg/L)"] / res["Ca teoritis (mg/L)"] * 100).fillna(0),
             marker="o", capsize=4, color="C2")
ax2.axhline(100, color="gray", ls="--", lw=1)
ax2.set_xlabel("Konsentrasi pengganggu (mg/L)")
ax2.set_ylabel("% Recovery Ca")
ax2.grid(alpha=0.3)
st.pyplot(fig2)

with st.expander("Rumus yang digunakan"):
    st.markdown(r"""
- Kalibrasi (OLS): $A = mC + b$, $R^2 = 1 - SS_{res}/SS_{tot}$
- Konsentrasi sampel: $C = \dfrac{A - b}{m} \times FP$ (dihitung per ulangan, lalu dirata-rata)
- SD & %RSD memakai $n-1$ (SD sampel): $\%RSD = SD/\bar{C}\times 100$
- % Recovery $= \bar{C}_{terukur}/C_{teoritis}\times 100$
- % Gangguan $= (\bar{C}_{sampel}-\bar{C}_{kontrol})/\bar{C}_{kontrol}\times 100$
  (negatif = penekanan sinyal, positif = peningkatan sinyal)
- Signifikansi: uji t Welch dua sisi terhadap kontrol
- LOD $= 3{,}3\,s/m$, LOQ $= 10\,s/m$ ($s$ = SD blanko bila diisi, jika tidak $S_{y/x}$ regresi)
""")
