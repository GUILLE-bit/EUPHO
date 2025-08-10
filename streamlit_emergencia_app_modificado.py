
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

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

    def tansig(self, x):
        return np.tanh(x)

    def normalize_input(self, X_real):
        return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1

    def desnormalize_output(self, y_norm, ymin=-1, ymax=1):
        return (y_norm - ymin) / (ymax - ymin)

    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalize_output(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm)
        valor_max_emeac = 8.05
        emer_ac = emerrel_cumsum / valor_max_emeac
        emerrel_diff = np.diff(emer_ac, prepend=0)

        def clasificar(valor):
            if valor < 0.02:
                return "Bajo"
            elif valor <= 0.079:
                return "Medio"
            else:
                return "Alto"

        riesgo = np.array([clasificar(v) for v in emerrel_diff])

        return pd.DataFrame({
            "EMERREL(0-1)": emerrel_diff,
            "Nivel_Emergencia_relativa": riesgo
        })

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)

# ------------------ Interfaz Streamlit ------------------
st.title("Predicción de Emergencia Agrícola con ANN")

st.info(
    "**Formato requerido del archivo Excel:**\n"
    "- Columnas: `Julian_days`, `TMAX`, `TMIN`, `Prec`\n"
    "- El día 1 debe corresponder al **1 de septiembre de 2025**.\n"
    "- Asegúrate de que los nombres de las columnas estén escritos exactamente igual."
)


st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%",
    min_value=0.5,
    max_value=2.84,
    value=1.75,
    step=0.01,
    format="%.2f"
)

uploaded_files = st.file_uploader(
    "Sube uno o más archivos Excel (.xlsx) con columnas: Julian_days, TMAX, TMIN, Prec",
    type=["xlsx"],
    accept_multiple_files=True
)

modelo = PracticalANNModel()

def obtener_colores(niveles):
    return niveles.map({"Bajo": "green", "Medio": "orange", "Alto": "red"})

legend_labels = [
    plt.Line2D([0], [0], color='green', lw=4, label='Bajo'),
    plt.Line2D([0], [0], color='orange', lw=4, label='Medio'),
    plt.Line2D([0], [0], color='red', lw=4, label='Alto')
]

# NUEVO rango de fechas
fecha_inicio = pd.to_datetime("2025-09-01")
fecha_fin = pd.to_datetime("2026-03-01")

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_excel(file)
        if not all(col in df.columns for col in ["Julian_days", "TMAX", "TMIN", "Prec"]):
            st.warning(f"{file.name} no tiene las columnas requeridas.")
            continue

        X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()
        fechas = pd.to_datetime("2025-09-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")

        pred = modelo.predict(X_real)
        pred["Fecha"] = fechas
        pred["Julian_days"] = df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
        pred["EMEAC (0-1)"] = pred["EMERREL acumulado"] / umbral_usuario
        pred["EMEAC (%)"] = pred["EMEAC (0-1)"] * 100

        nombre = Path(file.name).stem
        colores = obtener_colores(pred["Nivel_Emergencia_relativa"])

        st.subheader(f"EMERREL (0-1) - {nombre}")
        # Calcular media móvil de 5 días
        pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
        fig_er, ax_er = plt.subplots(figsize=(14, 5), dpi=150)
        ax_er.bar(pred["Fecha"], pred["EMERREL(0-1)"], color=colores, label="EMERREL (0-1)")
        ax_er.plot(pred["Fecha"], pred["EMERREL_MA5"], color="blue", linewidth=2.2, label="Media móvil 5 días")
        ax_er.set_title(f"Emergencia Relativa Diaria - {nombre}")
        ax_er.set_xlabel("Fecha")
        ax_er.set_ylabel("EMERREL (0-1)")
        ax_er.grid(True, linestyle="--", alpha=0.5)
        ax_er.set_xlim(fecha_inicio, fecha_fin)
        ax_er.legend(handles=legend_labels + [plt.Line2D([0], [0], color="blue", lw=2, label="Media móvil 5 días")], title="Niveles")
        ax_er.xaxis.set_major_locator(mdates.MonthLocator())
        ax_er.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax_er.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig_er)

        st.subheader(f"EMEAC (%) - {nombre}")
        fechas_validas = pd.to_datetime(pred["Fecha"])
        emeac_pct = pd.to_numeric(pred["EMEAC (%)"], errors="coerce")
        validez = ~(fechas_validas.isna() | emeac_pct.isna())
        fechas_plot = fechas_validas[validez].to_numpy()
        emeac_plot = emeac_pct[validez].to_numpy()

        fig_eac, ax_eac = plt.subplots(figsize=(14, 5), dpi=150)
        ax_eac.fill_between(fechas_plot, emeac_plot, color="skyblue", alpha=0.5)
        ax_eac.plot(fechas_plot, emeac_plot, color="blue", linewidth=2)

        niveles = [25, 50, 75, 90]
        colores_niveles = ['gray', 'green', 'orange', 'red']
        for nivel, color in zip(niveles, colores_niveles):
            ax_eac.axhline(nivel, linestyle='--', color=color, linewidth=1.5, label=f'{nivel}%')

        ax_eac.set_title(f"Progreso EMEAC (%) - {nombre} (Umbral: {umbral_usuario})")
        ax_eac.set_xlabel("Fecha")
        ax_eac.set_ylabel("EMEAC (%)")
        ax_eac.set_ylim(0, 100)
        ax_eac.set_xlim(fecha_inicio, fecha_fin)
        ax_eac.grid(True, linestyle="--", alpha=0.5)
        ax_eac.xaxis.set_major_locator(mdates.MonthLocator())
        ax_eac.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_eac.legend(title="Niveles EMEAC (%)")
        plt.setp(ax_eac.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig_eac)

        st.subheader(f"Datos calculados - {nombre}")
        tabla_filtrada = pred[["Fecha", "Nivel_Emergencia_relativa", "EMEAC (%)"]]
        st.dataframe(tabla_filtrada)
        csv = tabla_filtrada.to_csv(index=False).encode("utf-8")
        st.download_button(f"Descargar CSV - {nombre}", csv, f"{nombre}_EMEAC.csv", "text/csv")
else:
    st.info("Sube al menos un archivo .xlsx para iniciar el análisis.")
