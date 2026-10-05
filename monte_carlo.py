                (exit_gate_rules, False),
            ):
                for gate, share in rules[(terminal, cargo)]:
                    if float(share) <= 0:
                        continue
                    for name in names:
                        g, n = gate_index[gate], node_index[name]
                        start, end = (gate, name) if is_entry else (name, gate)
                        target = entry_route if is_entry else exit_route
                        cell = (g, n) if is_entry else (n, g)
                        if np.isnan(target[cell]):
                            try:
                                target[cell] = _route_time_min(
                                    model, graph, start, end,
                                    average_speed_kmh, junctions, overrides,
                                )
                            except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
                                raise RuntimeError(
                                    f"No route from {start} to {end}."
                                ) from exc
    entry_route = np.nan_to_num(entry_route)
    exit_route = np.nan_to_num(exit_route)

    crossing = getattr(model, "INTERNAL_GATE_CROSSING_SEC", {})
    terminal_min = np.array([
        [
            float(terminal_process_min[terminal][cargo])
            + float(crossing.get(terminal, 0.0)) / 60.0
            for cargo in ("FULL", "EMPTY")
        ]
        for terminal in terminals
    ])

    service_min = (
        np.array([
            float(gate_times_sec["FULL_ENTRY"]),
            float(gate_times_sec["EMPTY_ENTRY"]),
        ]) / 60.0,
        np.array([
            float(gate_times_sec["FULL_EXIT"]),
            float(gate_times_sec["EMPTY_EXIT"]),
        ]) / 60.0,
    )

    cfg = {
        "trucks_per_day": trucks_per_day,
        "arrival_variability": arrival_variability,
        "demand_variability": demand_variability,
        "management_enabled": management_enabled,
        "max_entries_per_hour": max_entries_per_hour,
        "terminals": terminals,
        "gates": gates,
        "terminal_shares": [float(terminal_shares[t]) for t in terminals],
        "cargo_shares": [
            [float(cargo_shares[t]["FULL"]), float(cargo_shares[t]["EMPTY"])]
            for t in terminals
        ],
        "entry_cum": entry_cum,
        "exit_cum": exit_cum,
        "dest_cum": dest_cum,
        "dest_node": dest_node,
        "entry_route_min": entry_route,
        "exit_route_min": exit_route,
        "terminal_min": terminal_min,
        "service_min": service_min,
        "entry_lanes": {
            g: int(gate_lanes.get(g, {}).get("entry", 0)) for g in gates
        },
        "exit_lanes": {
            g: int(gate_lanes.get(g, {}).get("exit", 0)) for g in gates
        },
        # main.py snapshots queues every 5 minutes until 24:00.
        "snapshot_times": np.arange(
            0.0, 24.0 * 60.0 + 1e-9, QUEUE_SNAPSHOT_INTERVAL_MIN
        ),
    }

    # --------------------------------------------------------
    # Run batches
    # --------------------------------------------------------
    if batch_size is None:
        batch_size = max(1, MAX_BATCH_CELLS // max(trucks_per_day, 1))
    batch_size = int(min(batch_size, simulations))

    rng = np.random.default_rng(int(seed))
    frames = []
    done = 0
    while done < simulations:
        batch = min(batch_size, simulations - done)
        entry_kpis, exit_kpis = _simulate_batch(cfg, rng, batch)
        frames.append(
            _batch_rows(cfg, entry_kpis, exit_kpis, batch, done + 1, seed)
        )
        done += batch
        if progress_callback is not None:
            progress_callback(done, simulations)

    df = pd.concat(frames, ignore_index=True)

    metric_columns = [
        c for c in df.columns
        if c.startswith(("peak_queue_", "avg_wait_", "peak_wait_"))
        and c != "peak_queue_hour"
    ]
    metrics = _percentile_summary(df, metric_columns)

    # Representative scenario: closest to the median peak queue.
    target = float(df["peak_queue_all"].median())
    idx = (df["peak_queue_all"] - target).abs().idxmin()
    representative = df.loc[int(idx)].to_dict()

    gate_capacities = {}
    for gate in gates:
        for operation, lanes in (
            ("ENTRY", cfg["entry_lanes"][gate]),
            ("EXIT", cfg["exit_lanes"][gate]),
        ):
            gate_capacities[f"{gate} {operation}"] = {
                "lanes": lanes,
                "configured": True,
            }

    return {
        "results": df,
        "metrics": metrics,
        "representative": representative,
        "gate_capacities_vph": gate_capacities,
        "network_metrics": {},
        "network_percentiles": {},
        "spillback_metrics": {},
        "spillback_percentiles": {},
        "queue_metrics": metrics,
        "queue_percentiles": metrics,
        "configuration": {
            "simulations": simulations,
            "workers": 1,
            "batch_size": batch_size,
            "model_type": "gate_cycle",
            "trucks_per_day": trucks_per_day,
            "arrival_variability": arrival_variability,
            "demand_variability": demand_variability,
            "management_enabled": management_enabled,
            "max_entries_per_hour": max_entries_per_hour,
            "seed": int(seed),
            "roads_simulated": False,
            "junctions_simulated": False,
        },
    }
