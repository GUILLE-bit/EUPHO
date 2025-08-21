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
EMEAC_MAX_DEN = 2.10

# ------------------- Modelo ANN con pesos embebidos ---------------------
class PracticalANNModel:
    def __init__(self):
        self.IW = np.array([...])  # <<-- recortado para simplificar
        self.bias_IW = np.array([...])
        self.LW = np.array([...])
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
st.title("Predicción de Emergencia Agrícola con ANN")

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral ajustable de EMEAC para 100%",
    min_value=0.5, max_value=2.84, value=1.75, step=0.01, format="%.2f"
)

uploaded_files = st.file_uploader("Sube Excel con Julian_days, TMAX, TMIN, Prec", type=["xlsx"], accept_multiple_files=True)
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
        pred["EMEAC (0-1) - mínimo"] = pred["EMERREL acumulado"] / EMEAC_MIN_DEN
        pred["EMEAC (0-1) - máximo"] = pred["EMERREL acumulado"] / EMEAC_MAX_DEN
        pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
        for col in ["EMEAC (0-1) - mínimo","EMEAC (0-1) - máximo","EMEAC (0-1) - ajustable"]:
            pred[col.replace("(0-1)", "(%)")] = (pred[col]*100).clip(0,100)
        pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(5,1).mean()
        pred_vis = pred[(pred["Fecha"]>=fecha_inicio)&(pred["Fecha"]<=fecha_fin)].copy()

        nombre = Path(file.name).stem

        # --- Gráfico 1 ---
        st.subheader("EMERGENCIA RELATIVA DIARIA - BORDENAVE")
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
                             hovermode="x unified", height=650, width=1200)
        fig_er.update_xaxes(range=[fecha_inicio, fecha_fin], dtick="M1", tickformat="%b")
        st.plotly_chart(fig_er, theme="streamlit")

        # --- Gráfico 2 ---
        st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - máximo"],
                                 mode="lines", line=dict(width=0), name="Máximo"))
        fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - mínimo"],
                                 mode="lines", line=dict(width=0), fill="tonexty", name="Mínimo"))
        fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - ajustable"],
                                 mode="lines", line=dict(width=2.5), name=f"Ajustable (/{umbral_usuario:.2f})"))
        fig.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)",
                          yaxis=dict(range=[0,100]), hovermode="x unified", height=600, width=1200)
        fig.update_xaxes(range=[fecha_inicio, fecha_fin], dtick="M1", tickformat="%b")
        st.plotly_chart(fig, theme="streamlit")

        # --- Tabla ---
        st.subheader(f"Resultados (sep → mar) - {nombre}")
        nivel_icono={"Bajo":"🟢 Bajo","Medio":"🟠 Medio","Alto":"🔴 Alto"}
        tabla=pred_vis[["Fecha","Julian_days","Nivel_Emergencia_relativa"]].copy()
        tabla["EMEAC (%)"]=pred_vis["EMEAC (%) - ajustable"]
        tabla["Nivel_Emergencia_relativa"]=tabla["Nivel_Emergencia_relativa"].map(nivel_icono)
        tabla=tabla.rename(columns={"Nivel_Emergencia_relativa":"Nivel de EMERREL"})
        st.dataframe(tabla, use_container_width=True)
        st.download_button("Descargar CSV", tabla.to_csv(index=False).encode("utf-8"),
                           f"{nombre}_resultados.csv","text/csv")
else:
    st.info("Sube al menos un archivo .xlsx para iniciar el análisis.")
