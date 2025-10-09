# import io
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st
# from scipy.signal import savgol_filter

# st.set_page_config(page_title="SPR Dual-Panel", layout="wide")

# # =============== Utilitas I/O & Prepro ===============

# def safe_read_table(file, skiprows=0):
#     try:
#         if hasattr(file, 'read'):
#             raw = file.read()
#             buf = io.BytesIO(raw)
#         else:
#             buf = file
#         df = pd.read_csv(buf, skiprows=skiprows)
#         return df
#     except Exception:
#         try:
#             for sep in ["\t", ";", "|", "\s+"]:
#                 if hasattr(file, 'read'):
#                     buf = io.BytesIO(raw)
#                 else:
#                     buf = file
#                 df = pd.read_csv(buf, sep=sep, engine='python', skiprows=skiprows)
#                 return df
#         except Exception:
#             return None

# def normalisasi(x):
#     x = np.asarray(x, dtype=float)
#     xmin, xmax = np.nanmin(x), np.nanmax(x)
#     if np.isclose(xmax - xmin, 0):
#         return np.zeros_like(x)
#     return (x - xmin) / (xmax - xmin)

# def xy_from_ratio_df(df, col_theta, col_v2, col_v1, drop_last=True):
#     try:
#         y_raw = (df[col_v2] / df[col_v1])
#         x = df[col_theta]
#     except KeyError as e:
#         raise KeyError(f"Kolom tidak ditemukan: {e}.")
#     if drop_last:
#         x = x.iloc[:-1]
#         y_raw = y_raw.iloc[:-1]
#     y = normalisasi(y_raw)
#     return np.asarray(x, float), np.asarray(y, float)

# def xy_from_fit_df(df, col_xfit, col_yfit):
#     try:
#         x = np.asarray(df[col_xfit], float)
#         y = np.asarray(df[col_yfit], float)
#         return x, y
#     except KeyError as e:
#         raise KeyError(f"Kolom tidak ditemukan: {e}.")

# def smooth_savgol(y, window_length=71, polyorder=2):
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

# def argmin_xy(x, y):
#     if len(y) == 0:
#         return np.nan, np.nan
#     i = int(np.nanargmin(y))
#     return float(x[i]), float(y[i])

# # =============== Sidebar ===============

# st.sidebar.title("Pengaturan Data")

# fit_files = st.sidebar.file_uploader("File kurva teori (CSV/TXT)", type=["csv", "txt", "dat"], accept_multiple_files=True)
# exp_files = st.sidebar.file_uploader("File eksperimen (CSV/TXT)", type=["csv", "txt", "dat"], accept_multiple_files=True)

# col_theta_name = st.sidebar.text_input("Kolom sudut (θ)", value="Sudut Motor 1")
# col_v2_name = st.sidebar.text_input("Kolom V2", value=" Tegangan Sensor 2")
# col_v1_name = st.sidebar.text_input("Kolom V1", value=" Tegangan Sensor 1")
# col_xfit_name = st.sidebar.text_input("Kolom x_fit", value="x_fit")
# col_yfit_name = st.sidebar.text_input("Kolom y_pred_local", value="y_pred_local")

# use_smooth = st.sidebar.checkbox("Aktifkan smoothing", value=True)
# window_len = st.sidebar.slider("window_length", 5, 201, 71, 2)
# poly_order = st.sidebar.slider("polyorder", 2, 5, 2)

# xmin, xmax = st.sidebar.slider("Rentang x (°)", 30.0, 70.0, (40.0, 48.0), 0.1)
# ymin_left, ymax_left = st.sidebar.slider("Y kiri (fit)", 0.0, 1.5, (0.0, 0.9), 0.05)
# ymin_right, ymax_right = st.sidebar.slider("Y kanan (eksperimen)", 0.0, 1.5, (0.0, 1.0), 0.05)

# zoom_xmin, zoom_xmax = st.sidebar.slider("x zoom", 30.0, 70.0, (43.5, 45.5), 0.05)
# zoom_ymin, zoom_ymax = st.sidebar.slider("y zoom", 0.0, 1.0, (0.01, 0.75), 0.01)

# st.title("Visualisasi SPR: Fit vs Eksperimen (High DPI + Label)")

# fit_curves, exp_curves = [], []
# colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

# # =============== Proses file teori dan eksperimen ===============

# if fit_files:
#     for i, f in enumerate(fit_files):
#         df = safe_read_table(f)
#         if df is not None:
#             x, y = xy_from_fit_df(df, col_xfit_name, col_yfit_name)
#             fit_curves.append({"label": f.name, "x": x, "y": y, "color": colors[i % len(colors)]})

# if exp_files:
#     for j, f in enumerate(exp_files):
#         df = safe_read_table(f, skiprows=1)
#         if df is not None:
#             x, y = xy_from_ratio_df(df, col_theta_name, col_v2_name, col_v1_name)
#             if use_smooth:
#                 y = smooth_savgol(y, window_length=window_len, polyorder=poly_order)
#             exp_curves.append({"label": f.name, "x": x, "y": y, "color": colors[j % len(colors)]})

# # =============== Plotting (High DPI + Label) ===============

# fig, (ax_right,ax_left) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
# plt.style.use('seaborn-v0_8-whitegrid')

# # Panel kiri (teori)
# for c in fit_curves:
#     ax_left.plot(c["x"], c["y"], linewidth=2.2, color=c["color"], label=c["label"])
#     xm, ym = argmin_xy(c["x"], c["y"])
#     ax_left.text(xm, ym + 0.02, f"{c['label']} ({xm:.2f}°)", fontsize=8, color=c["color"], ha='center')

# ax_left.set_title("Kurva Teori/Fit SPR")
# ax_left.set_xlabel("θᵢ (°)")
# ax_left.set_ylabel("Rasio Tegangan (teori)")
# ax_left.set_xlim(xmin, xmax)
# ax_left.set_ylim(ymin_left, ymax_left)
# ax_left.grid(True, linestyle='--', linewidth=0.5)
# ax_left.legend(fontsize=8)

# # Inset zoom kiri
# inset = ax_left.inset_axes([0.62, 0.08, 0.35, 0.42])
# for c in fit_curves:
#     inset.plot(c["x"], c["y"], linewidth=1.8, color=c["color"])
#     xm, ym = argmin_xy(c["x"], c["y"])
#     inset.vlines(xm, zoom_ymin, ym, color=c["color"], linestyle='--')
#     inset.text(xm, ym + 0.01, f"{xm:.2f}°", fontsize=7, color=c["color"], ha='center')

# inset.set_xlim(zoom_xmin, zoom_xmax)
# inset.set_ylim(zoom_ymin, zoom_ymax)
# inset.grid(True, linestyle=':')
# ax_left.indicate_inset_zoom(inset, edgecolor="black")

# # Panel kanan (eksperimen)
# for c in exp_curves:
#     ax_right.plot(c["x"], c["y"], linewidth=2.0, color=c["color"], label=c["label"])
#     xm, ym = argmin_xy(c["x"], c["y"])
#     ax_right.text(xm, ym + 0.02, f"{c['label']} ({xm:.2f}°)", fontsize=8, color=c["color"], ha='center')

# ax_right.set_title("Kurva Eksperimen SPR")
# ax_right.set_xlabel("θᵢ (°)")
# ax_right.set_ylabel("Rasio Tegangan (norm.)")
# ax_right.set_xlim(xmin, xmax)
# ax_right.set_ylim(ymin_right, ymax_right)
# ax_right.grid(True, linestyle='--', linewidth=0.5)
# ax_right.legend(fontsize=8)

# plt.tight_layout()
# st.pyplot(fig)

# # =============== Analitik Minima ===============

# st.subheader("Analitik Minima")
# col1, col2 = st.columns(2)
# with col1:
#     if fit_curves:
#         df_fit = pd.DataFrame([{ 'Label': c['label'], 'θ_min (°)': argmin_xy(c['x'], c['y'])[0], 'R_min': argmin_xy(c['x'], c['y'])[1] } for c in fit_curves])
#         st.dataframe(df_fit, width='stretch')
#     else:
#         st.info("Belum ada kurva teori.")
# with col2:
#     if exp_curves:
#         df_exp = pd.DataFrame([{ 'Label': c['label'], 'θ_min (°)': argmin_xy(c['x'], c['y'])[0], 'R_min': argmin_xy(c['x'], c['y'])[1] } for c in exp_curves])
#         st.dataframe(df_exp, width='stretch')
#     else:
#         st.info("Belum ada kurva eksperimen.")


# app_streamlit_spr.py
# -------------------------------------------------------------
# Versi revisi: hanya menambahkan teks label (tanpa label legend), high DPI.
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import savgol_filter

st.set_page_config(page_title="SPR Dual-Panel", layout="wide")

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

st.title("Visualisasi SPR: Fit vs Eksperimen (High DPI + Minima Markers)")

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

# =============== Plotting (High DPI + Text Label only) ===============

fig, (ax_right, ax_left) = plt.subplots(1, 2, figsize=(18, 7), dpi=500)
plt.style.use('seaborn-v0_8-whitegrid')

# Panel kiri (teori)
for c in fit_curves:
    ax_left.plot(c["x"], c["y"], linewidth=2.2, color=c["color"])  # tanpa label legend

ax_left.set_title("Kurva Teori/Fit SPR")
ax_left.set_xlabel("θᵢ (°)")
ax_left.set_ylabel("Rasio Tegangan (teori)")
ax_left.set_xlim(xmin, xmax)
ax_left.set_ylim(ymin_left, ymax_left)
ax_left.grid(True, linestyle='--', linewidth=0.5)

# Inset zoom kiri — tambah garis vertikal putus-putus + legenda derajat
inset = ax_left.inset_axes([0.60, 0.08, 0.38, 0.44])

minima_lines = []
minima_labels = []
for c in fit_curves:
    inset.plot(c["x"], c["y"], linewidth=1.8, color=c["color"])  # plot kurva
    xm, ym = argmin_xy(c["x"], c["y"])  # posisi minima
    # garis vertikal putus-putus di posisi minima
    line = inset.vlines(xm, zoom_ymin, zoom_ymax, color=c["color"], linestyle='--', linewidth=1.2)
    minima_lines.append(line)
    minima_labels.append(f"{xm:.2f}°")
    # beri anotasi kecil di dekat titik minima
    inset.scatter([xm], [ym], s=18, color=c["color"])
    inset.text(xm + 0.02, ym + 0.02, f"{xm:.2f}°", fontsize=7, color=c["color"], ha='left', va='bottom')

# Set rentang zoom
inset.set_xlim(zoom_xmin, zoom_xmax)
inset.set_ylim(zoom_ymin, zoom_ymax)
inset.grid(True, linestyle=':')

# legenda ringan berisi hanya angka derajat (tanpa nama kurva)
try:
    inset.legend(minima_lines, minima_labels, loc='upper right', fontsize=8, frameon=False)
except Exception:
    pass

ax_left.indicate_inset_zoom(inset, edgecolor="black")

# Panel kanan (eksperimen) — hanya kurva + tanpa legend; tidak ada label samping
for c in exp_curves:
    ax_right.plot(c["x"], c["y"], linewidth=2.0, color=c["color"])  

ax_right.set_title("Kurva Eksperimen SPR")
ax_right.set_xlabel("θᵢ (°)")
ax_right.set_ylabel("Rasio Tegangan (norm.)")
ax_right.set_xlim(xmin, xmax)
ax_right.set_ylim(ymin_right, ymax_right)
ax_right.grid(True, linestyle='--', linewidth=0.5)

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
