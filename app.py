# import io
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st
# from scipy.signal import savgol_filter

# st.set_page_config(page_title="SPR Dual-Panel", layout="wide")

# # =============== Utilitas I/O & Prepro ===============

# def safe_read_table(file, skiprows=0):
#     """Baca CSV/TXT dengan pandas. Coba deteksi delimiter otomatis.
#     Mengembalikan DataFrame atau None jika gagal.
#     """
#     try:
#         # Jika file adalah UploadedFile Streamlit
#         if hasattr(file, 'read'):
#             raw = file.read()
#             buf = io.BytesIO(raw)
#         else:
#             # path string
#             buf = file
#         # coba baca dengan sep default
#         df = pd.read_csv(buf, skiprows=skiprows)
#         return df
#     except Exception:
#         try:
#             # Coba delimiter umum
#             for sep in ["\t", ";", "|", "\s+"]:
#                 if hasattr(file, 'read'):
#                     buf = io.BytesIO(raw)
#                 else:
#                     buf = file
#                 df = pd.read_csv(buf, sep=sep, engine='python', skiprows=skiprows)
#                 return df
#         except Exception:
#             return None


# def normalisasi(x: pd.Series | np.ndarray):
#     x = np.asarray(x, dtype=float)
#     xmin, xmax = np.nanmin(x), np.nanmax(x)
#     if np.isclose(xmax - xmin, 0):
#         return np.zeros_like(x)
#     return (x - xmin) / (xmax - xmin)


# def xy_from_ratio_df(df: pd.DataFrame,
#                      col_theta: str,
#                      col_v2: str,
#                      col_v1: str,
#                      drop_last: bool = True):
#     """Ambil sumbu x = sudut, y = (V2/V1) lalu normalisasi seperti di skrip asli."""
#     try:
#         y_raw = (df[col_v2] / df[col_v1])
#         x = df[col_theta]
#     except KeyError as e:
#         raise KeyError(f"Kolom tidak ditemukan: {e}. Pastikan pemetaan kolom benar.")
#     if drop_last:
#         x = x.iloc[:-1]
#         y_raw = y_raw.iloc[:-1]
#     y = normalisasi(y_raw)
#     return np.asarray(x, float), np.asarray(y, float)


# def xy_from_fit_df(df: pd.DataFrame, col_xfit: str, col_yfit: str):
#     """Ambil x_fit, y_pred_local untuk kurva teori/fitting."""
#     try:
#         x = np.asarray(df[col_xfit], float)
#         y = np.asarray(df[col_yfit], float)
#         return x, y
#     except KeyError as e:
#         raise KeyError(f"Kolom tidak ditemukan: {e}. Pastikan pemetaan kolom benar.")


# def smooth_savgol(y: np.ndarray, window_length: int = 71, polyorder: int = 2):
#     # jaga supaya window ganjil dan cukup besar
#     if window_length % 2 == 0:
#         window_length += 1
#     if window_length < polyorder + 2:
#         window_length = polyorder + 3
#         if window_length % 2 == 0:
#             window_length += 1
#     if y.size < window_length:
#         return y
#     try:
#         return savgol_filter(y, window_length, polyorder)
#     except Exception:
#         return y


# def argmin_xy(x: np.ndarray, y: np.ndarray):
#     if len(y) == 0:
#         return np.nan, np.nan
#     i = int(np.nanargmin(y))
#     return float(x[i]), float(y[i])


# # =============== Sidebar: Upload & Konfigurasi ===============

# st.sidebar.title("Pengaturan Data")

# st.sidebar.markdown("**1) Upload kurva teori/fitting (CSV/TXT)** — bisa lebih dari satu.")
# fit_files = st.sidebar.file_uploader(
#     "File kurva teori (mis. local_curve.csv)", type=["csv", "txt", "dat"], accept_multiple_files=True
# )

# st.sidebar.markdown("**2) Upload data eksperimen (CSV/TXT)** — bisa lebih dari satu.")
# exp_files = st.sidebar.file_uploader(
#     "File eksperimen (berisi sudut & tegangan sensor)", type=["csv", "txt", "dat"], accept_multiple_files=True
# )

# st.sidebar.divider()

# st.sidebar.markdown("**Pemetaan Kolom (Eksperimen)**")
# def_text_theta = "Sudut Motor 1"
# def_text_v2 = " Tegangan Sensor 2"  # perhatikan spasi depan sesuai file asli
# ndef_text_v1 = " Tegangan Sensor 1"
# col_theta_name = st.sidebar.text_input("Nama kolom sudut (θ)", value=def_text_theta)
# col_v2_name = st.sidebar.text_input("Nama kolom V2", value=def_text_v2)
# col_v1_name = st.sidebar.text_input("Nama kolom V1", value=ndef_text_v1)

# st.sidebar.markdown("**Pemetaan Kolom (Teori/Fit)**")
# col_xfit_name = st.sidebar.text_input("Nama kolom x_fit", value="x_fit")
# col_yfit_name = st.sidebar.text_input("Nama kolom y_pred_local", value="y_pred_local")

# st.sidebar.divider()

# st.sidebar.markdown("**Smoothing (Eksperimen)**")
# use_smooth = st.sidebar.checkbox("Aktifkan smoothing Savitzky–Golay", value=True)
# window_len = st.sidebar.slider("window_length (ganjil)", min_value=5, max_value=201, value=71, step=2)
# poly_order = st.sidebar.slider("polyorder", min_value=2, max_value=5, value=2, step=1)

# st.sidebar.divider()

# st.sidebar.markdown("**Rentang Plot**")
# xmin, xmax = st.sidebar.slider("Batas x (°)", min_value=30.0, max_value=70.0, value=(40.0, 48.0), step=0.1)
# ymin_left, ymax_left = st.sidebar.slider("Batas y kiri (fit)", min_value=0.0, max_value=1.5, value=(0.0, 0.9), step=0.05)
# ymin_right, ymax_right = st.sidebar.slider("Batas y kanan (eksperimen)", min_value=0.0, max_value=1.5, value=(0.0, 1.0), step=0.05)

# st.sidebar.divider()

# st.sidebar.markdown("**Inset Zoom (panel kiri)**")
# zoom_xmin, zoom_xmax = st.sidebar.slider("x zoom (°)", min_value=30.0, max_value=70.0, value=(43.5, 45.5), step=0.05)
# zoom_ymin, zoom_ymax = st.sidebar.slider("y zoom", min_value=0.0, max_value=1.0, value=(0.01, 0.75), step=0.01)

# # =============== Parsing file menjadi kurva ===============

# st.title("Visualisasi SPR: Fit vs Data Eksperimen (Dual Panel)")

# with st.expander("Petunjuk Singkat", expanded=False):
#     st.markdown(
#         """
#         - Unggah **kurva teori/fitting** (kolom default: `x_fit`, `y_pred_local`).
#         - Unggah **data eksperimen** (kolom default: `Sudut Motor 1`, ` Tegangan Sensor 2`, ` Tegangan Sensor 1`).
#         - Aktifkan smoothing bila perlu (data eksperimen saja), atur **window_length** dan **polyorder**.
#         - Panel kiri menampilkan kurva fit; panel kanan menampilkan data eksperimen yang dinormalisasi.
#         - Garis vertikal pada inset kiri menandai posisi **minima** masing-masing kurva.
#         """
#     )

# # Kumpulan kurva teori: list of dict
# fit_curves = []  # {label, x, y, color}
# exp_curves = []  # {label, x, y, color}

# palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

# # --------- proses kurva teori/fitting ---------
# if fit_files:
#     for i, f in enumerate(fit_files):
#         df = safe_read_table(f)
#         if df is None:
#             st.warning(f"Gagal membaca file teori: {getattr(f, 'name', f)}")
#             continue
#         try:
#             x, y = xy_from_fit_df(df, col_xfit_name, col_yfit_name)
#             label = f"Teori/Fit #{i+1} — {getattr(f, 'name', 'file')}"
#             color = palette[i % len(palette)] if palette else None
#             fit_curves.append({"label": label, "x": x, "y": y, "color": color})
#         except KeyError as e:
#             st.error(str(e))

# # --------- proses data eksperimen ---------
# if exp_files:
#     for j, f in enumerate(exp_files):
#         # DATAxx.TXT di skrip asli menggunakan skiprows=1
#         df = safe_read_table(f, skiprows=1)
#         if df is None:
#             st.warning(f"Gagal membaca file eksperimen: {getattr(f, 'name', f)}")
#             continue
#         try:
#             x, y = xy_from_ratio_df(df, col_theta=col_theta_name, col_v2=col_v2_name, col_v1=col_v1_name)
#             if use_smooth:
#                 y = smooth_savgol(y, window_length=window_len, polyorder=poly_order)
#             label = f"Eksperimen #{j+1} — {getattr(f, 'name', 'file')}"
#             color = palette[j % len(palette)] if palette else None
#             exp_curves.append({"label": label, "x": x, "y": y, "color": color})
#         except KeyError as e:
#             st.error(str(e))

# # =============== Plotting ===============

# fig, (ax_right,ax_left) = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
# plt.style.use('seaborn-v0_8-whitegrid')

# # Panel kiri: teori/fitting
# for c in fit_curves:
#     ax_left.plot(c["x"], c["y"], label=c["label"], linewidth=2.2, color=c["color"])
# ax_left.set_title("Grafik Fitting Reflektansi SPR")
# # ax_left.set_xlabel(r"$\\theta_i$ (°)")
# ax_left.set_xlabel("θᵢ (°)")

# ax_left.set_ylabel("Rasio Tegangan (teori)")
# ax_left.set_xlim(xmin, xmax)
# ax_left.set_ylim(ymin_left, ymax_left)
# ax_left.grid(True, linestyle='--', linewidth=0.5)
# ax_left.legend(fontsize=9)

# # Inset zoom di panel kiri
# inset = ax_left.inset_axes([0.62, 0.08, 0.35, 0.42])
# for c in fit_curves:
#     inset.plot(c["x"], c["y"], linewidth=2.0, color=c["color"])
#     xm, ym = argmin_xy(c["x"], c["y"])
#     if not np.isnan(xm):
#         inset.vlines(x=xm, ymin=zoom_ymin, ymax=ym, color=c["color"], linestyle='--')
#         inset.text(xm + 0.03, ym + 0.02, f"{xm:.2f}°", fontsize=8)
#         # garis bantu tipis di panel utama (opsional)
#         ax_left.axvline(x=xm, ymin=0, ymax=1, color=c["color"], alpha=0.15, linestyle=':')

# inset.set_xlim(zoom_xmin, zoom_xmax)
# inset.set_ylim(zoom_ymin, zoom_ymax)
# inset.grid(True, linestyle=':')
# ax_left.indicate_inset_zoom(inset, edgecolor="black")

# # Panel kanan: eksperimen
# for c in exp_curves:
#     ax_right.plot(c["x"], c["y"], label=c["label"], linewidth=2.0, color=c["color"])
# ax_right.set_title("Grafik Data Pengukuran SPR")
# # ax_right.set_xlabel(r"$\\theta_i$ (°)")
# ax_right.set_xlabel("θᵢ (°)")
# ax_right.set_ylabel("Rasio Tegangan (ternormalisasi)")
# ax_right.set_xlim(xmin, xmax)
# ax_right.set_ylim(ymin_right, ymax_right)
# ax_right.grid(True, linestyle='--', linewidth=0.5)
# ax_right.legend(fontsize=9)

# plt.tight_layout()
# st.pyplot(fig)

# # =============== Analitik Ringkas (Delta R & Delta Theta) ===============

# st.subheader("Analitik Minima (opsional)")
# colA, colB = st.columns(2)
# with colA:
#     st.markdown("**Teori/Fit** — posisi minima ")
#     if fit_curves:
#         df_fit = pd.DataFrame([
#             {
#                 "Label": c["label"],
#                 "theta_min (°)": argmin_xy(c["x"], c["y"])[0],
#                 "R_min": argmin_xy(c["x"], c["y"])[1],
#             }
#             for c in fit_curves
#         ])
#         st.dataframe(df_fit, width='stretch')
#     else:
#         st.info("Belum ada kurva teori diunggah.")

# with colB:
#     st.markdown("**Eksperimen** — posisi minima ")
#     if exp_curves:
#         df_exp = pd.DataFrame([
#             {
#                 "Label": c["label"],
#                 "theta_min (°)": argmin_xy(c["x"], c["y"])[0],
#                 "R_min": argmin_xy(c["x"], c["y"])[1],
#             }
#             for c in exp_curves
#         ])
#         st.dataframe(df_exp, width='stretch')
#     else:
#         st.info("Belum ada kurva eksperimen diunggah.")

# st.markdown(
#     """
#     *Catatan:* `ΔR` dapat dihitung relatif terhadap baseline (mis. kurva emas saja) dengan mengambil perbedaan nilai `R_min`.
#     `Δθ` dapat dihitung sebagai selisih `theta_min` antar-kurva yang ingin dibandingkan.
#     Silakan ekspor tabel di atas dan lakukan perhitungan lanjutan sesuai kebutuhan.
#     """
# )



# app_streamlit_spr.py
# -------------------------------------------------------------
# Aplikasi Web Streamlit untuk visualisasi kurva SPR (baseline & hybrid)
# Versi ini menambahkan label teks di tiap kurva dan meningkatkan resolusi (dpi) grafik.
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import savgol_filter

st.set_page_config(page_title="SPR Dual-Panel", layout="wide")

# =============== Utilitas I/O & Prepro ===============

def safe_read_table(file, skiprows=0):
    try:
        if hasattr(file, 'read'):
            raw = file.read()
            buf = io.BytesIO(raw)
        else:
            buf = file
        df = pd.read_csv(buf, skiprows=skiprows)
        return df
    except Exception:
        try:
            for sep in ["\t", ";", "|", "\s+"]:
                if hasattr(file, 'read'):
                    buf = io.BytesIO(raw)
                else:
                    buf = file
                df = pd.read_csv(buf, sep=sep, engine='python', skiprows=skiprows)
                return df
        except Exception:
            return None

def normalisasi(x):
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if np.isclose(xmax - xmin, 0):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def xy_from_ratio_df(df, col_theta, col_v2, col_v1, drop_last=True):
    try:
        y_raw = (df[col_v2] / df[col_v1])
        x = df[col_theta]
    except KeyError as e:
        raise KeyError(f"Kolom tidak ditemukan: {e}.")
    if drop_last:
        x = x.iloc[:-1]
        y_raw = y_raw.iloc[:-1]
    y = normalisasi(y_raw)
    return np.asarray(x, float), np.asarray(y, float)

def xy_from_fit_df(df, col_xfit, col_yfit):
    try:
        x = np.asarray(df[col_xfit], float)
        y = np.asarray(df[col_yfit], float)
        return x, y
    except KeyError as e:
        raise KeyError(f"Kolom tidak ditemukan: {e}.")

def smooth_savgol(y, window_length=71, polyorder=2):
    if window_length % 2 == 0:
        window_length += 1
    if window_length < polyorder + 2:
        window_length = polyorder + 3
        if window_length % 2 == 0:
            window_length += 1
    if y.size < window_length:
        return y
    try:
        return savgol_filter(y, window_length, polyorder)
    except Exception:
        return y

def argmin_xy(x, y):
    if len(y) == 0:
        return np.nan, np.nan
    i = int(np.nanargmin(y))
    return float(x[i]), float(y[i])

# =============== Sidebar ===============

st.sidebar.title("Pengaturan Data")

fit_files = st.sidebar.file_uploader("File kurva teori (CSV/TXT)", type=["csv", "txt", "dat"], accept_multiple_files=True)
exp_files = st.sidebar.file_uploader("File eksperimen (CSV/TXT)", type=["csv", "txt", "dat"], accept_multiple_files=True)

col_theta_name = st.sidebar.text_input("Kolom sudut (θ)", value="Sudut Motor 1")
col_v2_name = st.sidebar.text_input("Kolom V2", value=" Tegangan Sensor 2")
col_v1_name = st.sidebar.text_input("Kolom V1", value=" Tegangan Sensor 1")
col_xfit_name = st.sidebar.text_input("Kolom x_fit", value="x_fit")
col_yfit_name = st.sidebar.text_input("Kolom y_pred_local", value="y_pred_local")

use_smooth = st.sidebar.checkbox("Aktifkan smoothing", value=True)
window_len = st.sidebar.slider("window_length", 5, 201, 71, 2)
poly_order = st.sidebar.slider("polyorder", 2, 5, 2)

xmin, xmax = st.sidebar.slider("Rentang x (°)", 30.0, 70.0, (40.0, 48.0), 0.1)
ymin_left, ymax_left = st.sidebar.slider("Y kiri (fit)", 0.0, 1.5, (0.0, 0.9), 0.05)
ymin_right, ymax_right = st.sidebar.slider("Y kanan (eksperimen)", 0.0, 1.5, (0.0, 1.0), 0.05)

zoom_xmin, zoom_xmax = st.sidebar.slider("x zoom", 30.0, 70.0, (43.5, 45.5), 0.05)
zoom_ymin, zoom_ymax = st.sidebar.slider("y zoom", 0.0, 1.0, (0.01, 0.75), 0.01)

st.title("Visualisasi SPR: Fit vs Eksperimen (High DPI + Label)")

fit_curves, exp_curves = [], []
colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

# =============== Proses file teori dan eksperimen ===============

if fit_files:
    for i, f in enumerate(fit_files):
        df = safe_read_table(f)
        if df is not None:
            x, y = xy_from_fit_df(df, col_xfit_name, col_yfit_name)
            fit_curves.append({"label": f.name, "x": x, "y": y, "color": colors[i % len(colors)]})

if exp_files:
    for j, f in enumerate(exp_files):
        df = safe_read_table(f, skiprows=1)
        if df is not None:
            x, y = xy_from_ratio_df(df, col_theta_name, col_v2_name, col_v1_name)
            if use_smooth:
                y = smooth_savgol(y, window_length=window_len, polyorder=poly_order)
            exp_curves.append({"label": f.name, "x": x, "y": y, "color": colors[j % len(colors)]})

# =============== Plotting (High DPI + Label) ===============

fig, (ax_right,ax_left) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid')

# Panel kiri (teori)
for c in fit_curves:
    ax_left.plot(c["x"], c["y"], linewidth=2.2, color=c["color"], label=c["label"])
    xm, ym = argmin_xy(c["x"], c["y"])
    ax_left.text(xm, ym + 0.02, f"{c['label']} ({xm:.2f}°)", fontsize=8, color=c["color"], ha='center')

ax_left.set_title("Kurva Teori/Fit SPR")
ax_left.set_xlabel("θᵢ (°)")
ax_left.set_ylabel("Rasio Tegangan (teori)")
ax_left.set_xlim(xmin, xmax)
ax_left.set_ylim(ymin_left, ymax_left)
ax_left.grid(True, linestyle='--', linewidth=0.5)
ax_left.legend(fontsize=8)

# Inset zoom kiri
inset = ax_left.inset_axes([0.62, 0.08, 0.35, 0.42])
for c in fit_curves:
    inset.plot(c["x"], c["y"], linewidth=1.8, color=c["color"])
    xm, ym = argmin_xy(c["x"], c["y"])
    inset.vlines(xm, zoom_ymin, ym, color=c["color"], linestyle='--')
    inset.text(xm, ym + 0.01, f"{xm:.2f}°", fontsize=7, color=c["color"], ha='center')

inset.set_xlim(zoom_xmin, zoom_xmax)
inset.set_ylim(zoom_ymin, zoom_ymax)
inset.grid(True, linestyle=':')
ax_left.indicate_inset_zoom(inset, edgecolor="black")

# Panel kanan (eksperimen)
for c in exp_curves:
    ax_right.plot(c["x"], c["y"], linewidth=2.0, color=c["color"], label=c["label"])
    xm, ym = argmin_xy(c["x"], c["y"])
    ax_right.text(xm, ym + 0.02, f"{c['label']} ({xm:.2f}°)", fontsize=8, color=c["color"], ha='center')

ax_right.set_title("Kurva Eksperimen SPR")
ax_right.set_xlabel("θᵢ (°)")
ax_right.set_ylabel("Rasio Tegangan (norm.)")
ax_right.set_xlim(xmin, xmax)
ax_right.set_ylim(ymin_right, ymax_right)
ax_right.grid(True, linestyle='--', linewidth=0.5)
ax_right.legend(fontsize=8)

plt.tight_layout()
st.pyplot(fig)

# =============== Analitik Minima ===============

st.subheader("Analitik Minima")
col1, col2 = st.columns(2)
with col1:
    if fit_curves:
        df_fit = pd.DataFrame([{ 'Label': c['label'], 'θ_min (°)': argmin_xy(c['x'], c['y'])[0], 'R_min': argmin_xy(c['x'], c['y'])[1] } for c in fit_curves])
        st.dataframe(df_fit, width='stretch')
    else:
        st.info("Belum ada kurva teori.")
with col2:
    if exp_curves:
        df_exp = pd.DataFrame([{ 'Label': c['label'], 'θ_min (°)': argmin_xy(c['x'], c['y'])[0], 'R_min': argmin_xy(c['x'], c['y'])[1] } for c in exp_curves])
        st.dataframe(df_exp, width='stretch')
    else:
        st.info("Belum ada kurva eksperimen.")
