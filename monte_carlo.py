"""Fast uncertainty screening for the Jeddah Islamic Port traffic model.

This module intentionally does NOT run 10,000 full SimPy replications. It runs
an analytical hourly network screening that preserves the same terminal/cargo/
gate/path logic as the main model and reports gate queue pressure plus network
V/C uncertainty.
"""
import math
import numpy as np
import pandas as pd
import networkx as nx


def _normalise(d):
    total = float(sum(d.values()))
    if total <= 0:
        raise ValueError("Probability weights must sum to a positive value.")
    return {k: float(v) / total for k, v in d.items()}


def _profile_24h(model):
    minute = np.asarray(model.create_base_hourly_profile(), dtype=float)
    hourly = np.array([minute[h * 60:(h + 1) * 60].sum() for h in range(24)], dtype=float)
    hourly /= hourly.sum()
    return hourly


def _entry_probabilities(model, entry_rules):
    probs = {g: 0.0 for g in model.GATE_LANES}
    for terminal, t_share in model.TERMINAL_SHARES.items():
        for cargo, c_share in model.CARGO_SHARES[terminal].items():
            for gate, share in entry_rules[(terminal, cargo)]:
                probs[gate] = probs.get(gate, 0.0) + t_share * c_share * share
    return _normalise(probs)


def _exit_probabilities(model, exit_rules):
    probs = {g: 0.0 for g in model.GATE_LANES}
    for terminal, t_share in model.TERMINAL_SHARES.items():
        for cargo, c_share in model.CARGO_SHARES[terminal].items():
            for gate, share in exit_rules[(terminal, cargo)]:
                probs[gate] = probs.get(gate, 0.0) + t_share * c_share * share
    return _normalise(probs)


def _candidate_paths(graph, start, end, model, route_overrides):
    override = (route_overrides or {}).get((start, end))
    if override:
        return [(list(override), 1.0)]

    try:
        generator = nx.shortest_simple_paths(
            graph, source=start, target=end, weight="travel_time_min"
        )
        paths = []
        for path in generator:
            paths.append(path)
            if len(paths) >= max(1, int(model.ROUTE_ALTERNATIVE_COUNT)):
                break
    except nx.NetworkXNoPath as exc:
        raise RuntimeError(f"No route found from {start} to {end}") from exc

    if not paths:
        raise RuntimeError(f"No route found from {start} to {end}")
    if model.ROUTE_RANDOMIZATION <= 0 or len(paths) == 1:
        return [(paths[0], 1.0)]

    costs = np.array([
        sum(graph[a][b]["travel_time_min"] for a, b in zip(path[:-1], path[1:]))
        for path in paths
    ], dtype=float)
    min_cost = float(costs.min())
    alt_weights = np.array([max(0.05, min_cost / c) for c in costs[1:]], dtype=float)
    if alt_weights.sum() <= 0:
        return [(paths[0], 1.0)]
    alt_share = float(model.ROUTE_RANDOMIZATION)
    probs = [1.0 - alt_share]
    probs.extend((alt_share * alt_weights / alt_weights.sum()).tolist())
    return list(zip(paths, probs))


def _build_flow_options(model, graph, access_rules, exit_rules, route_overrides, logistics_destination_daily=None):
    """Return weighted path options used by the fast Monte Carlo layer."""
    options = []
    logistics_destination_daily = logistics_destination_daily or model.LOGISTICS_DESTINATION_DAILY
    for terminal, t_share in model.TERMINAL_SHARES.items():
        for cargo, c_share in model.CARGO_SHARES[terminal].items():
            destinations = (list(model.MPT_DESTINATION_SPLIT.keys()) if terminal == "MPT"
                else list(logistics_destination_daily.keys()) if terminal == "LOGISTICS"
                else [None])
            for mpt_dest in destinations:
                if terminal == "MPT": dest_share=model.MPT_DESTINATION_SPLIT[mpt_dest]
                elif terminal == "LOGISTICS": dest_share=float(getattr(model, "LOGISTICS_DESTINATION_SHARE", {}).get(mpt_dest, 0.0))
                else: dest_share=1.0
                destination=model.get_destination_node(terminal,mpt_dest)
                for entry_gate, entry_share in access_rules[(terminal, cargo)]:
                    for exit_gate, exit_share in exit_rules[(terminal, cargo)]:
                        base_weight = t_share * c_share * dest_share * entry_share * exit_share
                        if base_weight <= 0:
                            continue
                        entry_paths = _candidate_paths(graph, entry_gate, destination, model, route_overrides)
                        exit_paths = _candidate_paths(graph, destination, exit_gate, model, route_overrides)
                        for ep, ep_prob in entry_paths:
                            for xp, xp_prob in exit_paths:
                                options.append({
                                    "weight": base_weight * ep_prob * xp_prob,
                                    "terminal": terminal,
                                    "cargo": cargo,
                                    "entry_gate": entry_gate,
                                    "exit_gate": exit_gate,
                                    "entry_path": ep,
                                    "exit_path": xp,
                                    "terminal_time_min": model.TERMINAL_PROCESS_MIN[terminal][cargo],
                                })

    total = sum(x["weight"] for x in options)
    if total <= 0:
        raise ValueError("No positive traffic flow options could be constructed.")
    for x in options:
        x["weight"] /= total
    return options


def _link_index(options):
    links = {}
    for opt in options:
        for path_name in ("entry_path", "exit_path"):
            for a, b in zip(opt[path_name][:-1], opt[path_name][1:]):
                links.setdefault((a, b), len(links))
    return links


def _gate_capacity(model, gate, operation, gate_lanes, rules, access_rules=None, gate_times_sec=None):
    lanes = int(gate_lanes.get(gate, {}).get(operation.lower(), 0))
    times = gate_times_sec or model.GATE_TIMES_SEC
    if lanes <= 0:
        return 0.0
    weights=[]
    if operation == "ENTRY":
        for terminal,t_share in model.TERMINAL_SHARES.items():
            for cargo,c_share in model.CARGO_SHARES[terminal].items():
                for gg,share in (access_rules or model.ENTRY_ACCESS_RULES)[(terminal, cargo)]:
                    if gg==gate:
                        key = "FULL_ENTRY" if cargo == "FULL" else "EMPTY_ENTRY"
                        weights.append((times[key], t_share*c_share*share))
    else:
        for terminal,t_share in model.TERMINAL_SHARES.items():
            for cargo,c_share in model.CARGO_SHARES[terminal].items():
                for gg,share in rules[(terminal,cargo)]:
                    if gg==gate:
                        key = "FULL_EXIT" if cargo == "FULL" else "EMPTY_EXIT"
                        weights.append((times[key], t_share*c_share*share))
    total=sum(w for _,w in weights)
    if total<=0: return 0.0
    avg_sec=sum(sec*w for sec,w in weights)/total
    return lanes*3600.0/avg_sec if avg_sec>0 else 0.0


def run_monte_carlo(
    model,
    simulations=10_000,
    trucks_per_day=10_500,
    arrival_variability=0.05,
    demand_variability=0.03,
    management_enabled=False,
    max_entries_per_hour=600,
    seed=42,
    entry_gate_rules=None,
    exit_gate_rules=None,
    entry_access_rules=None,
    gate_lanes=None,
    route_overrides=None,
    route_randomization=None,
    logistics_destination_daily=None,
    gate_times_sec=None,
    n2_n1_corridor_lanes=None,
    lanes_per_direction=None,
    capacity_per_lane_vph=None,
    terminal_shares=None,
    cargo_shares=None,
):
    """Fast hourly Monte Carlo screening.

    The previous implementation only modelled hourly *entry gate* pressure,
    which can legitimately return zero queue under the current assumptions and
    ignored the dashboard's edited route/gate settings. This version also
    propagates sampled flow through the internal network and reports peak
    directional network V/C.
    """
    rng = np.random.default_rng(seed)

    entry_rules = entry_gate_rules or model.ENTRY_GATE_RULES
    exit_rules = exit_gate_rules or model.EXIT_GATE_RULES
    access_rules = entry_access_rules or model.ENTRY_ACCESS_RULES
    gate_lanes = gate_lanes or model.GATE_LANES
    route_overrides=route_overrides or {}
    logistics_destination_daily={k:int(v) for k,v in (logistics_destination_daily or model.LOGISTICS_DESTINATION_DAILY).items()}
    if terminal_shares is not None:
        model.TERMINAL_SHARES = {str(k): float(v) for k,v in terminal_shares.items()}
    if cargo_shares is not None:
        model.CARGO_SHARES = {str(k): {str(c): float(v) for c,v in shares.items()} for k,shares in cargo_shares.items()}

    old_terminal_shares = dict(model.TERMINAL_SHARES)
    old_cargo_shares = {k:dict(v) for k,v in model.CARGO_SHARES.items()}
    old_randomization = model.ROUTE_RANDOMIZATION
    old_corridor_lanes = model.N2_N1_CORRIDOR_LANES
    old_gate_times = model.GATE_TIMES_SEC
    if n2_n1_corridor_lanes is not None:
        model.N2_N1_CORRIDOR_LANES = int(n2_n1_corridor_lanes)
    if gate_times_sec is not None:
        model.GATE_TIMES_SEC = gate_times_sec
    if route_randomization is not None:
        model.ROUTE_RANDOMIZATION = float(route_randomization)

    try:
        routes=model.read_kml_routes()
        graph=model.build_network(routes)
        flow_options=_build_flow_options(model,graph,access_rules,exit_rules,route_overrides,logistics_destination_daily=logistics_destination_daily)
        link_ids = _link_index(flow_options)
        link_names = list(link_ids.keys())
        default_link_capacity = float((lanes_per_direction or model.LANES_PER_DIRECTION) *
                                      (capacity_per_lane_vph or model.CAPACITY_PER_LANE_VPH))
        link_capacity = np.array([
            float(graph[a][b].get("capacity_vph", default_link_capacity))
            for a, b in link_names
        ], dtype=float)

        option_probs = np.array([x["weight"] for x in flow_options], dtype=float)
        entry_gates = list(gate_lanes.keys())
        exit_gates = list(gate_lanes.keys())
        entry_p = _entry_probabilities(model, entry_rules)
        exit_p = _exit_probabilities(model, exit_rules)
        entry_p_arr = np.array([entry_p.get(g, 0.0) for g in entry_gates], dtype=float)
        exit_p_arr = np.array([exit_p.get(g, 0.0) for g in exit_gates], dtype=float)

        entry_cap = {g: _gate_capacity(model, g, "ENTRY", gate_lanes, entry_rules, access_rules=access_rules, gate_times_sec=gate_times_sec) for g in entry_gates}
        exit_cap = {g: _gate_capacity(model, g, "EXIT", gate_lanes, exit_rules, gate_times_sec=gate_times_sec) for g in exit_gates}

        base = _profile_24h(model)
        records = []
        for sim in range(int(simulations)):
            daily = int(trucks_per_day)
            noise = np.exp(rng.normal(0, arrival_variability, 24))
            profile = base * noise
            profile /= profile.sum()
            desired = rng.multinomial(daily, profile)

            released = desired.copy()
            uncovered = 0
            if management_enabled:
                release_list=[]; backlog=0; h=0
                while h < 24 or backlog > 0:
                    scheduled = int(desired[h]) if h < 24 else 0
                    available = backlog + scheduled
                    rel = min(int(max_entries_per_hour), available)
                    release_list.append(rel)
                    backlog = available - rel
                    h += 1
                released = np.asarray(release_list, dtype=int)
                uncovered = backlog

            gate_q_in = np.zeros(len(entry_gates), dtype=float)
            gate_q_out = np.zeros(len(exit_gates), dtype=float)
            peak_gate = {g: 0.0 for g in gate_lanes}
            link_peak = np.zeros(len(link_names), dtype=float)
            weighted_wait = 0.0
            peak_spillback = 0.0
            total_released = int(released.sum())

            for h in range(len(released)):
                n = int(released[h])
                if n == 0:
                    continue
                counts = rng.multinomial(n, option_probs)

                entry_arr = np.zeros(len(entry_gates), dtype=int)
                exit_arr = np.zeros(len(exit_gates), dtype=int)
                link_flow = np.zeros(len(link_names), dtype=int)
                for i, count in enumerate(counts):
                    if count == 0:
                        continue
                    opt = flow_options[i]
                    entry_arr[entry_gates.index(opt["entry_gate"])] += count
                    exit_arr[exit_gates.index(opt["exit_gate"])] += count
                    for a, b in zip(opt["entry_path"][:-1], opt["entry_path"][1:]):
                        link_flow[link_ids[(a, b)]] += count
                    for a, b in zip(opt["exit_path"][:-1], opt["exit_path"][1:]):
                        link_flow[link_ids[(a, b)]] += count

                # Gate queue stock: separate IN/OUT resources, combined for KPI.
                for j, g in enumerate(entry_gates):
                    gate_q_in[j] = max(0.0, gate_q_in[j] + entry_arr[j] - entry_cap[g])
                    peak_gate[g] = max(peak_gate[g], gate_q_in[j])
                for j, g in enumerate(exit_gates):
                    gate_q_out[j] = max(0.0, gate_q_out[j] + exit_arr[j] - exit_cap[g])
                    peak_gate[g] = max(peak_gate[g], gate_q_out[j])

                link_vc = link_flow / np.maximum(link_capacity, 1e-9)
                link_peak = np.maximum(link_peak, link_vc)

                # Fast spillback screening: if an exit-gate queue exceeds the
                # physical storage of its final approach link, flag the
                # upstream network as blocked.
                for gate in exit_gates:
                    q = float(gate_q_out[exit_gates.index(gate)])
                    if q <= 0: continue
                    approach_candidates = [i for i, (a,b) in enumerate(link_names) if b == gate]
                    if approach_candidates:
                        idx = approach_candidates[0]
                        storage = max(1.0, float(graph[link_names[idx][0]][link_names[idx][1]].get("capacity_vph", 0.0)) * float(graph[link_names[idx][0]][link_names[idx][1]].get("travel_time_min", 0.0)) / 60.0)
                        excess = max(0.0, q - storage)
                        peak_spillback = max(peak_spillback, excess)
                        if excess > 0:
                            link_peak[idx] = max(link_peak[idx], 1.0 + excess / storage)
                weighted_wait += float(gate_q_in.sum() + gate_q_out.sum())

            peak_network_vc = float(link_peak.max()) if len(link_peak) else 0.0
            peak_link_idx = int(np.argmax(link_peak)) if len(link_peak) else 0
            peak_link = f"{link_names[peak_link_idx][0]} → {link_names[peak_link_idx][1]}" if link_names else ""
            gate_wait_proxy = weighted_wait / max(total_released, 1) * 60.0
            terminal_time = sum(
                model.TERMINAL_SHARES[t]
                * sum(model.CARGO_SHARES[t][c] * model.TERMINAL_PROCESS_MIN[t][c]
                      for c in model.CARGO_SHARES[t])
                for t in model.TERMINAL_SHARES
            )

            row = {
                "simulation": sim + 1,
                "daily_demand": daily,
                "released_entries": total_released,
                "uncovered_end": int(uncovered if management_enabled else 0),
                "peak_queue_all": max(peak_gate.values()) if peak_gate else 0.0,
                "peak_network_vc": peak_network_vc,
                "peak_network_link": peak_link,
                "peak_spillback_trucks": float(peak_spillback),
                "avg_turn_proxy_min": terminal_time + gate_wait_proxy,
            }
            for g in gate_lanes:
                row[f"peak_queue_{g}"] = peak_gate.get(g, 0.0)
            records.append(row)

        df = pd.DataFrame(records)
        metric_cols = ["peak_queue_all", "peak_network_vc", "peak_spillback_trucks", "avg_turn_proxy_min"] + [f"peak_queue_{g}" for g in gate_lanes]
        metrics = {}
        for col in metric_cols:
            metrics[col] = {p: float(df[col].quantile(float(p[1:]) / 100.0)) for p in ["P10", "P50", "P90", "P95", "P99"]}

        target = np.array([
            df["peak_queue_all"].median(),
            df["peak_network_vc"].median(),
            df["avg_turn_proxy_min"].median(),
        ])
        x = df[["peak_queue_all", "peak_network_vc", "avg_turn_proxy_min"]].to_numpy(dtype=float)
        scale = np.maximum(np.abs(target), 1.0)
        representative_idx = int(np.argmin((((x - target) / scale) ** 2).sum(axis=1)))
        representative = df.iloc[representative_idx].to_dict()

        return {
            "results": df,
            "metrics": metrics,
            "representative": representative,
            "gate_capacities_vph": {**{f"{g} IN": entry_cap[g] for g in entry_cap},
                                    **{f"{g} OUT": exit_cap[g] for g in exit_cap}},
            "gate_probabilities": {**{f"{g} IN": entry_p.get(g, 0.0) for g in entry_gates},
                                    **{f"{g} OUT": exit_p.get(g, 0.0) for g in exit_gates}},
            "network_capacity_vph": {
                f"{a} → {b}": float(link_capacity[i])
                for i, (a, b) in enumerate(link_names)
            },
            "network_links": link_names,
        }
    finally:
        model.ROUTE_RANDOMIZATION=old_randomization
        model.N2_N1_CORRIDOR_LANES=old_corridor_lanes
        model.GATE_TIMES_SEC=old_gate_times
        model.TERMINAL_SHARES=old_terminal_shares
        model.CARGO_SHARES=old_cargo_shares
