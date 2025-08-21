# ... resto de tu código sin cambios arriba ...

        # ===================== Gráfico 1: EMERGENCIA RELATIVA DIARIA (Plotly) =====================
        st.subheader("EMERGENCIA RELATIVA DIARIA - BORDENAVE")
        colores = pred_vis["Nivel_Emergencia_relativa"].map(COLOR_MAP).fillna(COLOR_FALLBACK).tolist()

        fig_er = go.Figure()
        fig_er.add_bar(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL(0-1)"],
            marker=dict(color=colores),
            customdata=pred_vis["Nivel_Emergencia_relativa"],
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
            name="EMERREL (0-1)"
        )
        fig_er.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL_MA5"],
            mode="lines",
            name="Media móvil 5 días",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
        ))
        fig_er.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL_MA5"],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(135, 206, 250, 0.3)",
            hoverinfo="skip",
            showlegend=False
        ))
        fig_er.add_hline(y=THR_BAJO_MEDIO, line_dash="dot", line_color=COLOR_MAP["Bajo"], annotation_text=f"Bajo ≤ {THR_BAJO_MEDIO:.3f}")
        fig_er.add_hline(y=THR_MEDIO_ALTO, line_dash="dot", line_color=COLOR_MAP["Medio"], annotation_text=f"Medio ≤ {THR_MEDIO_ALTO:.3f}")

        fig_er.update_layout(
            xaxis_title="Fecha",
            yaxis_title="EMERREL (0-1)",
            hovermode="x unified",
            legend_title="Referencias",
            height=650,   # altura original
            width=1200    # más ancho
        )
        fig_er.update_xaxes(range=[fecha_inicio, fecha_fin], dtick="M1", tickformat="%b")
        fig_er.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_er, theme="streamlit")

        # ===================== Gráfico 2: EMERGENCIA ACUMULADA DIARIA (Plotly) =====================
        st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE")
        fig = go.Figure()

        # Banda entre mínimo y máximo
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - máximo"],
            mode="lines",
            line=dict(width=0),
            name="Máximo",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - mínimo"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Mínimo",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
        ))

        # Línea de umbral ajustable
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - ajustable"],
            mode="lines",
            line=dict(width=2.5),
            name=f"Umbral ajustable (/{umbral_usuario:.2f})",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>"
        ))

        # Líneas horizontales 25, 50, 75, 90 %
        for nivel in [25, 50, 75, 90]:
            fig.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")

        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="EMEAC (%)",
            yaxis=dict(range=[0, 100]),
            hovermode="x unified",
            legend_title="Referencias",
            height=600,   # altura original
            width=1200    # más ancho
        )
        fig.update_xaxes(range=[fecha_inicio, fecha_fin], dtick="M1", tickformat="%b")
        st.plotly_chart(fig, theme="streamlit")

# ... resto de tu código sin cambios abajo ...
