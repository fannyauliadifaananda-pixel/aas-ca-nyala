import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

JUDUL = "Gangguan Analisis Ca"

# ============================================================ FUNGSI PERHITUNGAN
# Semua konsentrasi dalam mg/L (ppm). Absorbansi tanpa satuan.


def linreg(x, y):
    """Kalibrasi OLS: A = m*C + b, beserta statistik regresi."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("Minimal 3 titik kalibrasi.")
    if np.ptp(x) == 0:
        raise ValueError("Konsentrasi standar harus bervariasi.")
    m, b = np.polyfit(x, y, 1)
    ss_res = float(np.sum((y - (m * x + b)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sy_x = np.sqrt(ss_res / (n - 2))          # simpangan baku residual
    sxx = float(np.sum((x - x.mean()) ** 2))
    return dict(m=m, b=b, r2=r2, r=np.sign(m) * np.sqrt(max(r2, 0)), sy_x=sy_x,
                se_m=sy_x / np.sqrt(sxx), se_b=sy_x * np.sqrt(1 / n + x.mean() ** 2 / sxx),
                sxx=sxx, n=n, ybar=float(y.mean()))


def predict_conc(a, fit, k=1, alpha=0.05):
    """
    Konsentrasi dari absorbansi rata-rata (k ulangan) + ketidakpastian dari kurva:
      C0   = (A - b) / m
      s_C0 = (Sy/x / |m|) * sqrt(1/k + 1/n + (A - ybar)^2 / (m^2 * Sxx))
      U    = t(alpha/2, n-2) * s_C0
    """
    m, b = fit["m"], fit["b"]
    c0 = (a - b) / m
    s_c0 = (fit["sy_x"] / abs(m)) * np.sqrt(
        1 / k + 1 / fit["n"] + (a - fit["ybar"]) ** 2 / (m ** 2 * fit["sxx"]))
    t = stats.t.ppf(1 - alpha / 2, fit["n"] - 2)
    return c0, s_c0, t * s_c0


def lod_loq(m, sy_x=None, sd_blank=None):
    """LOD = 3*s/m ; LOQ = 10*s/m  (s = SD blanko bila ada, jika tidak Sy/x)."""
    s = sd_blank if (sd_blank is not None and sd_blank > 0) else sy_x
    return 3 * s / m, 10 * s / m


def analyze_interference(df, fit, blank=0.0, fp_default=1.0, alpha=0.05, batas=5.0):
    """
    df kolom: Perlakuan, Pengganggu (mg/L), Ca teoritis (mg/L), A1..An, FP
    Baris dengan Pengganggu = 0 (yang pertama) = kontrol (Ca saja).
    """
    m, b = fit["m"], fit["b"]
    a_cols = [c for c in df.columns if c.startswith("A") and c[1:].isdigit()]
    rows, reps = [], []
    for _, r in df.iterrows():
        a = pd.to_numeric(r[a_cols], errors="coerce").dropna().to_numpy(dtype=float) - blank
        if len(a) == 0:
            continue
        fp = pd.to_numeric(r.get("FP", fp_default), errors="coerce")
        fp = fp_default if pd.isna(fp) else float(fp)
        abar, k = a.mean(), len(a)
        c0, _, u = predict_conc(abar, fit, k, alpha)
        reps.append((a - b) / m * fp)                      # konsentrasi tiap ulangan
        rows.append({
            "Perlakuan": r["Perlakuan"],
            "Pengganggu (mg/L)": r["Pengganggu (mg/L)"],
            "Ca teoritis (mg/L)": r["Ca teoritis (mg/L)"],
            "n": k,
            "Rata-rata A": abar,
            "SD A": a.std(ddof=1) if k > 1 else np.nan,
            "%RSD A": a.std(ddof=1) / abs(abar) * 100 if (k > 1 and abar != 0) else np.nan,
            "Ca terukur (mg/L)": c0 * fp,
            "± U95 (mg/L)": u * fp,
        })
    if not rows:
        raise ValueError("Belum ada data absorbansi.")
    out = pd.DataFrame(rows)

    ctrl = out.index[pd.to_numeric(out["Pengganggu (mg/L)"], errors="coerce") == 0]
    ctrl = int(ctrl[0]) if len(ctrl) else 0
    a_ctrl = out.loc[ctrl, "Rata-rata A"]
    c_ctrl = out.loc[ctrl, "Ca terukur (mg/L)"]

    out["% Recovery"] = out["Ca terukur (mg/L)"] / out["Ca teoritis (mg/L)"] * 100
    out["% Gangguan (A)"] = (out["Rata-rata A"] - a_ctrl) / a_ctrl * 100 if a_ctrl else np.nan
    out["% Gangguan (C)"] = (out["Ca terukur (mg/L)"] - c_ctrl) / c_ctrl * 100 if c_ctrl else np.nan

    pv = []
    for i, c in enumerate(reps):
        if i == ctrl or len(c) < 2 or len(reps[ctrl]) < 2:
            pv.append(np.nan)
        else:
            pv.append(stats.ttest_ind(c, reps[ctrl], equal_var=False).pvalue)
    out["p (Welch)"] = pv

    status = []
    for i, p in enumerate(pv):
        if i == ctrl:
            status.append("Kontrol")
            continue
        besar = abs(out.loc[i, "% Gangguan (A)"]) > batas
        sig = True if np.isnan(p) else p < alpha
        status.append("Mengganggu" if (besar and sig) else "Tidak mengganggu")
    out["Status"] = status
    return out, ctrl


def kadar_sampel(c_baca, fp, volume_ml, massa_g):
    """Kadar (mg/kg) = C_baca(mg/L) x FP x V(mL) / massa(g)."""
    return c_baca * fp * volume_ml / massa_g


# ============================================================ TAMPILAN APLIKASI
st.set_page_config(page_title=JUDUL, page_icon="🧪", layout="wide")
st.title(f"🧪 {JUDUL}")
st.caption("Spektrofotometri Serapan Atom (AAS) nyala · alat 3D · Ca (mg/L)")

# ---------------------------------------------------------------- 1. KALIBRASI
st.header("1. Kurva Kalibrasi Ca")
st.write("Masukkan absorbansi standar Ca. Gunakan konsentrasi 0 untuk blanko.")

default_cal = pd.DataFrame({
    "Konsentrasi (mg/L)": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "Absorbansi": [0.002, 0.051, 0.100, 0.148, 0.199, 0.247],  # contoh, ganti data Anda
})
cal = st.data_editor(default_cal, num_rows="dynamic", use_container_width=True, key="cal")
cal = cal.apply(pd.to_numeric, errors="coerce").dropna()

blank_txt = st.text_input(
    "Absorbansi ulangan blanko pereaksi (opsional, pisahkan dengan koma)",
    placeholder="0.001, 0.002, 0.001, 0.003",
    help="Dipakai untuk LOD/LOQ dan (bila dicentang) koreksi blanko.")
subtract_blank = st.checkbox(
    "Koreksi semua absorbansi (standar dan sampel) dengan rata-rata blanko", value=False,
    help="Dikurangkan konsisten ke kurva kalibrasi DAN data gangguan/sampel. "
         "Tidak perlu bila blanko 0 mg/L sudah menjadi titik kurva.")

try:
    blank = None
    if blank_txt.strip():
        blank = np.array([float(v) for v in blank_txt.replace(";", ",").split(",") if v.strip()])
    a_blank = float(blank.mean()) if (subtract_blank and blank is not None and len(blank)) else 0.0
    x = cal["Konsentrasi (mg/L)"].to_numpy()
    y = cal["Absorbansi"].to_numpy() - a_blank
    fit = linreg(x, y)
except Exception as e:
    st.error(f"Kalibrasi belum valid: {e}")
    st.stop()

m, b = fit["m"], fit["b"]
if m <= 0:
    st.error("Slope ≤ 0: periksa data kalibrasi.")
    st.stop()

sd_blank = blank.std(ddof=1) if (blank is not None and len(blank) > 1) else None
lod, loq = lod_loq(m, fit["sy_x"], sd_blank)
c_char = 0.0044 / m  # konsentrasi karakteristik (sensitivitas, 1% serapan = A 0,0044)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Slope (m)", f"{m:.5f}")
c2.metric("Intersep (b)", f"{b:.5f}")
c3.metric("R²", f"{fit['r2']:.5f}")
c4.metric("n titik", fit["n"])
c5, c6, c7 = st.columns(3)
c5.metric("LOD (mg/L)", f"{lod:.3f}")
c6.metric("LOQ (mg/L)", f"{loq:.3f}")
c7.metric("Sensitivitas (mg/L per 1% serapan)", f"{c_char:.3f}")
st.latex(rf"A = {m:.5f}\,C \;{'+' if b >= 0 else '-'}\; {abs(b):.5f}")

if fit["r"] < 0.995:
    st.warning("Koefisien korelasi r < 0,995: linearitas kurang baik. "
               "Periksa standar yang menyimpang atau persempit rentang.")

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
    "Baris dengan **Pengganggu = 0** adalah kontrol (Ca saja). Isi absorbansi tiap ulangan "
    "(A1–A3), kosongkan sel yang tidak dipakai. **FP** = faktor pengenceran.")

default_int = pd.DataFrame({
    "Perlakuan": ["Kontrol (Ca saja)", "Ca + pengganggu 1", "Ca + pengganggu 2", "Ca + pengganggu 3"],
    "Pengganggu (mg/L)": [0.0, 10.0, 50.0, 100.0],
    "Ca teoritis (mg/L)": [3.0, 3.0, 3.0, 3.0],
    "A1": [0.148, 0.141, 0.128, 0.110],   # contoh, ganti data Anda
    "A2": [0.149, 0.140, 0.127, 0.111],
    "A3": [0.147, 0.142, 0.129, 0.109],
    "FP": [1.0, 1.0, 1.0, 1.0],
})
data = st.data_editor(default_int, num_rows="dynamic", use_container_width=True, key="int")

o1, o2 = st.columns(2)
alpha = o1.select_slider("Taraf signifikansi (α)", options=[0.01, 0.05, 0.10], value=0.05)
batas = o2.number_input("Batas gangguan yang diterima (±%)", min_value=0.0, value=5.0, step=1.0)

try:
    res, ctrl = analyze_interference(data, fit, a_blank, alpha=alpha, batas=batas)
except Exception as e:
    st.error(f"Data gangguan belum valid: {e}")
    st.stop()

st.subheader("Hasil")
fmt = {
    "Rata-rata A": "{:.4f}", "SD A": "{:.4f}", "%RSD A": "{:.2f}",
    "Ca terukur (mg/L)": "{:.3f}", "± U95 (mg/L)": "{:.3f}", "% Recovery": "{:.1f}",
    "% Gangguan (A)": "{:+.2f}", "% Gangguan (C)": "{:+.2f}", "p (Welch)": "{:.4f}",
}
st.dataframe(res.style.format(fmt, na_rep="-"), use_container_width=True)
st.download_button("⬇️ Unduh hasil (CSV)", res.to_csv(index=False).encode("utf-8"),
                   file_name="hasil_gangguan_ca.csv", mime="text/csv")

a_cols = [c for c in data.columns if c.startswith("A") and c[1:].isdigit()]
a_all = pd.to_numeric(data[a_cols].stack(), errors="coerce").dropna() - a_blank
if len(a_all) and (a_all.max() > y.max() * 1.05 or a_all.min() < y.min() - 0.005):
    st.warning("Ada absorbansi di luar rentang kalibrasi, hasil ekstrapolasi kurang andal. "
               "Encerkan sampel dan isi FP.")

g1, g2 = st.columns(2)
fig2, ax2 = plt.subplots(figsize=(5, 3.4))
ax2.plot(res["Pengganggu (mg/L)"], res["% Recovery"], marker="o", color="C2")
ax2.axhline(100, color="gray", ls="--", lw=1)
ax2.set_xlabel("Pengganggu (mg/L)")
ax2.set_ylabel("% Recovery Ca")
ax2.grid(alpha=0.3)
g1.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(5, 3.4))
ax3.plot(res["Pengganggu (mg/L)"], res["% Gangguan (A)"], marker="o", color="C1")
ax3.axhspan(-batas, batas, color="green", alpha=0.12, label=f"±{batas:g}% (diterima)")
ax3.axhline(0, color="gray", lw=1)
ax3.set_xlabel("Pengganggu (mg/L)")
ax3.set_ylabel("% Gangguan sinyal (A)")
ax3.legend()
ax3.grid(alpha=0.3)
g2.pyplot(fig3)

# ---------------------------------------------------------------- 3. SAMPEL
with st.expander("3. Kadar Ca dalam sampel & recovery spike (opsional)"):
    s1, s2, s3 = st.columns(3)
    massa = s1.number_input("Massa sampel (g)", min_value=0.0001, value=1.0, format="%.4f")
    vol = s2.number_input("Volume akhir larutan (mL)", min_value=0.1, value=50.0)
    fp_s = s3.number_input("Faktor pengenceran (FP)", min_value=1.0, value=1.0)
    t_smp = st.text_input("Absorbansi ulangan sampel (koma)", "0.090, 0.091, 0.089")
    t_spk = st.text_input("Absorbansi ulangan sampel + spike (koma, opsional)", "")
    c_add = st.number_input("Konsentrasi Ca spike dalam larutan terukur (mg/L)", min_value=0.0, value=0.0)
    try:
        a_s = np.array([float(v) for v in t_smp.replace(";", ",").split(",") if v.strip()]) - a_blank
        if len(a_s):
            c_s, _, u_s = predict_conc(a_s.mean(), fit, len(a_s), alpha)
            kd = kadar_sampel(c_s, fp_s, vol, massa)
            st.metric("Kadar Ca sampel (mg/kg)", f"{kd:.2f}",
                      help=f"C terbaca = {c_s:.3f} ± {u_s:.3f} mg/L")
            st.caption(f"C terbaca = {c_s:.3f} ± {u_s:.3f} mg/L · "
                       f"kadar ± {u_s * fp_s * vol / massa:.2f} mg/kg (95%)")
            if t_spk.strip() and c_add > 0:
                a_k = np.array([float(v) for v in t_spk.replace(";", ",").split(",") if v.strip()]) - a_blank
                c_k, _, _ = predict_conc(a_k.mean(), fit, len(a_k), alpha)
                rec = (c_k - c_s) / c_add * 100
                st.metric("% Recovery spike", f"{rec:.1f}%")
                st.caption("Rentang penerimaan umum: 80–120%.")
    except Exception as e:
        st.error(f"Input sampel belum valid: {e}")

with st.expander("Rumus yang digunakan"):
    st.markdown(r"""
- Kalibrasi (OLS): $A = mC + b$, $R^2 = 1 - SS_{res}/SS_{tot}$, $S_{y/x}=\sqrt{SS_{res}/(n-2)}$
- Koreksi blanko (bila dipilih): $A_{kor} = A - \bar{A}_{blanko}$, diterapkan ke standar **dan** sampel
- Konsentrasi: $C = \dfrac{\bar{A} - b}{m}\times FP$
- Ketidakpastian dari kurva (95%): $U = t_{(0,975;\,n-2)}\,\dfrac{S_{y/x}}{|m|}\sqrt{\dfrac1k+\dfrac1n+\dfrac{(\bar A-\bar y)^2}{m^2 S_{xx}}}$
- LOD $=3s/m$, LOQ $=10s/m$ ($s$ = SD blanko bila diisi, jika tidak $S_{y/x}$)
- Sensitivitas (konsentrasi karakteristik) $= 0{,}0044/m$ mg/L untuk 1% serapan
- % Recovery $= C_{terukur}/C_{teoritis}\times100$
- % Gangguan (A) $= (\bar A_{sampel}-\bar A_{kontrol})/\bar A_{kontrol}\times100$ — negatif = penekanan sinyal, positif = peningkatan
- Status "Mengganggu" bila $|\%\ gangguan| >$ batas **dan** signifikan (uji t Welch terhadap kontrol, $p<\alpha$)
- Kadar sampel (mg/kg) $= C\times FP\times V(mL)/massa(g)$
- Recovery spike $= (C_{sampel+spike}-C_{sampel})/C_{spike}\times100$
""")
