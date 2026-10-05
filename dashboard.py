
        # Use a fixed number of bins so the shape remains comparable across outputs.
        n_bins = 30
        vmin, vmax = float(values.min()), float(values.max())
        if np.isclose(vmin, vmax):
            edges = np.array([vmin - 0.5, vmax + 0.5])
        else:
            edges = np.linspace(vmin, vmax, n_bins + 1)
        counts, edges = np.histogram(values.to_numpy(), bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=centers,
            y=counts,
            width=widths * 0.92,
            marker_color="#4f8cff",
            marker_line_width=0,
            hovertemplate=f"{x_label}: %{{x:{value_format}}}<br>Frequency: %{{y}}<extra></extra>",
            name="Frequency",
        ))

        percentile_colors = {"P10":"#f59e0b","P50":"#7c8ea3","P90":"#94a3b8","P95":"#6ee7b7","P99":"#67e8f9"}
        for p, value in percentiles.items():
            fig.add_vline(
                x=value,
                line_width=2,
                line_dash="dash",
                line_color=percentile_colors[p],
                annotation_text=f"{p}: {value:{value_format}}",
                annotation_position="top",
                annotation_font_color=percentile_colors[p],
            )

        fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=25, b=20),
            paper_bgcolor="#12161c",
            plot_bgcolor="#12161c",
            font=dict(color="#f2f4f7"),
            xaxis=dict(title=x_label, gridcolor="#2a3038", zeroline=False),
            yaxis=dict(title="Frequency", gridcolor="#2a3038", zeroline=False),
            bargap=0.05,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        pct_df = pd.DataFrame({"Percentile": list(percentiles.keys()), "Value": list(percentiles.values())})
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, p in zip([c1,c2,c3,c4,c5], percentiles):
            col.metric(p, f"{percentiles[p]:{value_format}}{suffix}")

        selected_export = mc["results"][["simulation", "daily_demand", value_col]].copy()
        selected_export.columns = ["simulation", "daily_demand", selected_metric_label + (f" — {selected_gate}" if selected_gate else "")]
        csv_selected = selected_export.to_csv(index=False).encode("utf-8")
        csv_all = mc["results"].to_csv(index=False).encode("utf-8")
        csv_percentiles = pct_df.to_csv(index=False).encode("utf-8")
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Download selected distribution",
            csv_selected,
            "jeddah_monte_carlo_selected_output.csv",
            "text/csv",
            key="download_mc_selected",
        )
        d2.download_button(
            "Download all Monte Carlo results",
            csv_all,
            "jeddah_monte_carlo_all_results.csv",
            "text/csv",
            key="download_mc_all",
        )
        d3.download_button(
            "Download percentile summary",
            csv_percentiles,
            "jeddah_monte_carlo_percentiles.csv",
            "text/csv",
            key="download_mc_percentiles",
        )

st.subheader("Main junction bottlenecks")
st.caption("Node congestion is assessed from peak hourly movements through each junction relative to its configured effective capacity. Delay is shown separately because downstream spillback can create waiting even when node V/C remains below 1.0.")
if not junction_summary.empty:
    jc1,jc2,jc3=st.columns(3)
    for col,node in zip([jc1,jc2,jc3],["N01","N02","N03"]):
        row=junction_summary[junction_summary["junction"]==node]
        if row.empty:
            col.metric(node,"—")
            continue
        r=row.iloc[0]
        col.metric(node, f'{r["peak_vc"]:.2f} V/C', f'{r["congestion"]} | {int(r["peak_hourly_flow"]):,} vph')

    display_cols=[
        "junction","movements","peak_hour","peak_hourly_flow","capacity_vph",
        "peak_vc","utilization_pct","avg_wait_min","peak_wait_min",
        "peak_spillback_wait_min","congestion"
    ]
    display_cols=[c for c in display_cols if c in junction_summary.columns]
    js=junction_summary[display_cols].copy()
    js=js.rename(columns={
        "junction":"Node", "movements":"Movements", "peak_hour":"Peak hour",
        "peak_hourly_flow":"Peak flow (vph)", "capacity_vph":"Capacity (vph)",
        "peak_vc":"Peak V/C", "utilization_pct":"Peak utilisation (%)",
        "avg_wait_min":"Avg wait (min)", "peak_wait_min":"Peak wait (min)",
        "peak_spillback_wait_min":"Peak spillback wait (min)", "congestion":"Congestion"
    })
    st.dataframe(js,use_container_width=True,hide_index=True,
                 column_config={
                     "Peak V/C":st.column_config.NumberColumn(format="%.2f"),
                     "Peak utilisation (%)":st.column_config.NumberColumn(format="%.0f%%"),
                     "Avg wait (min)":st.column_config.NumberColumn(format="%.2f"),
                     "Peak wait (min)":st.column_config.NumberColumn(format="%.2f"),
                     "Peak spillback wait (min)":st.column_config.NumberColumn(format="%.2f"),
                 })
else:
    st.info("No junction summary was generated for this scenario.")

st.subheader("Junction movement detail")
if not junction_results.empty:
    js_detail=junction_results.groupby("junction").agg(movements=("truck_id","count"),peak_wait_min=("wait_min","max"),avg_wait_min=("wait_min","mean"),capacity_vph=("capacity_vph","first"),maneuver_delay_sec=("maneuver_delay_sec","first")).reset_index().sort_values("peak_wait_min",ascending=False)
    st.dataframe(js_detail,use_container_width=True,hide_index=True)
else: st.info("No junction movements were recorded in this scenario.")

st.subheader("Physical road network")
if not roads.empty:
    cols=[c for c in ["road_id","from_node","to_node","distance_km","peak_hourly_flow","capacity_vph","peak_vc","los","avg_wait_min"] if c in roads.columns]
    st.dataframe(roads[cols].sort_values("peak_vc",ascending=False),use_container_width=True)

st.subheader("Gate flows")
gate_summary=gates.groupby(["gate","operation"]).size().unstack(fill_value=0).reindex(GATES); st.dataframe(gate_summary,use_container_width=True)

st.subheader("Export results")
ec1,ec2,ec3=st.columns(3)
ec1.download_button("Download truck results",trucks.to_csv(index=False).encode("utf-8"),"jeddah_v1_trucks.csv","text/csv")
ec2.download_button("Download gate results",gates.to_csv(index=False).encode("utf-8"),"jeddah_v1_gates.csv","text/csv")
ec3.download_button("Download road results",roads.to_csv(index=False).encode("utf-8"),"jeddah_v1_roads.csv","text/csv")

st.subheader("Hourly arrivals")
trucks["arrival_hour"]=(trucks.arrival_min//60).astype(int); hourly=trucks.groupby("arrival_hour").size().reindex(range(24),fill_value=0); st.bar_chart(hourly)

with st.expander("Truck-level results"):
    display_cols=[c for c in ["truck_id","terminal","cargo","entry_gate","exit_gate","arrival_min","entry_path","exit_path","total_time_min","entry_gate_wait_min","exit_gate_wait_min"] if c in trucks.columns]
    st.dataframe(trucks[display_cols],use_container_width=True)
