import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# ====== Constantes de estilo/umbrales ======
THR_BAJO_MEDIO = 0.020
THR_MEDIO_ALTO = 0.079
COLOR_MAP = {"Bajo": "#2ca02c", "Medio": "#ff7f0e", "Alto": "#d62728"}
COLOR_FALLBACK = "#808080"

# Denominadores de EMEAC
EMEAC_MIN_DEN = 1.60
EMEAC_MAX_DEN = 2.00

# ------------------- Modelo ANN con pesos embebidos ---------------------
class PracticalANNModel:
    def __init__(self):
        self.IW = np.array([
            [-2.924160, -7.896739, -0.977000, 0.554961, 9.510761, 8.739410, 10.592497, 21.705275, -2.532038, 7.847811,
             -3.907758, 13.933289, 3.727601, 3.751941, 0.639185, -0.758034, 1.556183, 10.458917, -1.343551, -14.721089],
            [0.115434, 0.615363, -0.241457, 5.478775, -26.598709, -2.316081, 0.545053, -2.924576, -14.629911, -8.916969,
             3.516110, -6.315180, -0.005914, 10.801424, 4.928928, 1.158809, 4.394316, -23.519282, 2.694073, 3.387557],
            [6.210673, -0.666815, 2.923249, -8.329875, 7.029798, 1.202168, -4.650263, 2.243358, 22.006945, 5.118664,
             1.901176, -6.076520, 0.239450, -6.862627, -7.592373, 1.422826, -2.575074, 5.302610, -6.379549, -14.810670],
            [10.220671, 2.665316, 4.119266, 5.812964, -3.848171, 1.472373, -4.829068, -7.422444, 0.862384, 0.001028,
             0.853059, 2.953289, 1.403689, -3.040909, -6.946802, -1.799923, 0.994357, -5.551789, -0.764891, 5.520776]
        ])
        self.bias_IW = np.array([
            7.229977, -2.428431, 2.973525, 1.956296, -1.155897, 0.907013, 0.231416, 5.258464, 3.284862, 5.474901,
            2.971978, 4.302273, 1.650572, -1.768043, -7.693806, -0.010850, 1.497102, -2.799158, -2.366918, -9.754413
        ])
        self.LW = np.array([
            5.508609, -21.909052, -10.648533, -2.939799, 8.192068, -2.157424, -3.373238, -5.932938, -2.680237,
            -3.399422, 5.870659, -1.720078, 7.134293, 3.227154, -5.039080, -10.872101, -6.569051, -8.455429,
            2.703778, 4.776029
        ])
        self.bias_out = -5.394722
        self.input_min = np.array([1, 7.7, -3.5, 0])
        self.input_max = np.array([148, 38.5, 23.5, 59.9])

    def tansig(self, x): return np.tanh(x)
    def normalize_input(self, X_real): return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1
    def desnormalize_output(self, y_norm, ymin=-1, ymax=1): return (y_norm - ymin) / (ymax - ymin)
    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)
    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalize_output(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm)
        emer_ac = emerrel_cumsum / 8.05
        emerrel_diff = np.diff(emer_ac, prepend=0)
        def clasificar(v): return "Bajo" if v < 0.02 else ("Medio" if v <= 0.079 else "Alto")
        riesgo = np.array([clasificar(v) for v in emerrel_diff])
        return pd.DataFrame({"EMERREL(0-1)": emerrel_diff, "Nivel_Emergencia_relativa": riesgo})

# ------------------ Interfaz Streamlit ------------------
st.title("Predicción de Emergencia Agrícola EUPHO - NAPOSTA 2025")

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral ajustable de EMEAC para 100%",
    min_value=0.5, max_value=2.84, value=1.75, step=0.01, format="%.2f"
)

uploaded_files = st.file_uploader(
    "Sube Excel con Julian_days, TMAX, TMIN, Prec", type=["xlsx"], accept_multiple_files=True
)
modelo = PracticalANNModel()

fecha_inicio, fecha_fin = pd.to_datetime("2025-09-01"), pd.to_datetime("2026-03-01")

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_excel(file)
        if not all(c in df.columns for c in ["Julian_days", "TMAX", "TMIN", "Prec"]):
            st.warning(f"{file.name} no tiene columnas requeridas."); continue

        X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()
        fechas = pd.to_datetime("2025-09-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")

        pred = modelo.predict(X_real)
        pred["Fecha"], pred["Julian_days"] = fechas, df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()

        # Cálculos de EMEAC
        pred["EMEAC (0-1) - mínimo"]    = pred["EMERREL acumulado"] / EMEAC_MIN_DEN
        pred["EMEAC (0-1) - máximo"]    = pred["EMERREL acumulado"] / EMEAC_MAX_DEN
        pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
        for col in ["EMEAC (0-1) - mínimo", "EMEAC (0-1) - máximo", "EMEAC (0-1) - ajustable"]:
            pred[col.replace("(0-1)", "(%)")] = (pred[col] * 100).clip(0, 100)

        pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(5, 1).mean()
        pred_vis = pred[(pred["Fecha"] >= fecha_inicio) & (pred["Fecha"] <= fecha_fin)].copy()
        nombre = Path(file.name).stem

        # --- Gráfico 1 ---
        st.subheader("EMERGENCIA RELATIVA DIARIA - EUPHO - NAPOSTA 2025")
        colores = pred_vis["Nivel_Emergencia_relativa"].map(COLOR_MAP).fillna(COLOR_FALLBACK).tolist()
        fig_er = go.Figure()
        fig_er.add_bar(x=pred_vis["Fecha"], y=pred_vis["EMERREL(0-1)"],
                       marker=dict(color=colores), name="EMERREL (0-1)")
        fig_er.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5"],
                                    mode="lines", name="Media móvil 5 días"))
        fig_er.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5"],
                                    mode="lines", line=dict(width=0), fill="tozeroy",
                                    fillcolor="rgba(135,206,250,0.3)", showlegend=False, hoverinfo="skip"))
        fig_er.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
                             hovermode="x unified", height=650, width=1600)
        fig_er.update_xaxes(range=[fecha_inicio, fecha_fin], dtick="M1", tickformat="%b")
        st.plotly_chart(fig_er, theme="streamlit")

        # --- Gráfico 2 ---
        st.subheader("EMERGENCIA ACUMULADA DIARIA - EUPHO - NAPOSTA 2025")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - máximo"],
                                 mode="lines", line=dict(width=0), name="Máximo"))
        fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - mínimo"],
                                 mode="lines", line=dict(width=0), fill="tonexty", name="Mínimo"))
        fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - ajustable"],
                                 mode="lines", line=dict(width=2.5), name=f"Ajustable (/{umbral_usuario:.2f})"))
        fig.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)",
                          yaxis=dict(range=[0, 100]), hovermode="x unified", height=600, width=1600)
        fig.update_xaxes(range=[fecha_inicio, fecha_fin], dtick="M1", tickformat="%b")
        st.plotly_chart(fig, theme="streamlit")

        # --- Tabla ---
        st.subheader(f"Resultados (sep → mar) - {nombre}")
        nivel_icono = {"Bajo": "🟢 Bajo", "Medio": "🟠 Medio", "Alto": "🔴 Alto"}
        tabla = pred_vis[["Fecha", "Julian_days", "Nivel_Emergencia_relativa"]].copy()
        tabla["EMEAC (%)"] = pred_vis["EMEAC (%) - ajustable"]
        tabla["Nivel_Emergencia_relativa"] = tabla["Nivel_Emergencia_relativa"].map(nivel_icono)
        tabla = tabla.rename(columns={"Nivel_Emergencia_relativa": "Nivel de EMERREL"})
        st.dataframe(tabla, use_container_width=True)
        st.download_button("Descargar CSV", tabla.to_csv(index=False).encode("utf-8"),
                           f"{nombre}_resultados.csv", "text/csv")
else:
    st.info("Sube al menos un archivo .xlsx para iniciar el análisis.")
