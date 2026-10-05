"""Monte Carlo wrapper for the Jeddah Islamic Port traffic model.

IMPORTANT:
    Monte Carlo uses the SAME simulation engine as main.py.

Each realisation calls model.run_simulation(..., save_outputs=False).
The Monte Carlo layer only creates the realisation inputs and extracts KPIs;
it does not recreate gate queues, road capacity, routing, appointment
management, terminal processing or spillback with a second analytical model.

This makes the deterministic model and Monte Carlo directly comparable.
"""

from __future__ import annotations

import contextlib
import io
from copy import deepcopy

import numpy as np
import pandas as pd


PERCENTILES = {
    "P10": 0.10,
    "P50": 0.50,
    "P90": 0.90,
    "P95": 0.95,
    "P99": 0.99,
}


def _normalise_shares(values):
    values = {str(k): max(float(v), 0.0) for k, v in values.items()}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Shares must sum to a positive value.")
    return {k: v / total for k, v in values.items()}


def _perturb_shares(base, rng, variability):
    """Apply small controlled multiplicative variation and renormalise."""
    base = _normalise_shares(base)

    if variability <= 0:
        return base

    keys = list(base.keys())
    raw = np.array(
        [base[k] * np.exp(rng.normal(0.0, float(variability))) for k in keys],
        dtype=float,
    )
    raw /= raw.sum()
    return {k: float(v) for k, v in zip(keys, raw)}


def _build_realisation_shares(model, rng, variability, terminal_shares, cargo_shares):
    """Create one plausible terminal/cargo allocation around the configured scenario."""
    base_terminal = (
        terminal_shares
        if terminal_shares is not None
        else model.TERMINAL_SHARES
    )

    realised_terminal = _perturb_shares(
        base_terminal,
        rng,
        variability,
    )

    base_cargo = cargo_shares if cargo_shares is not None else model.CARGO_SHARES
    realised_cargo = {}

    for terminal, shares in base_cargo.items():
        realised_cargo[terminal] = _perturb_shares(
            shares,
            rng,
            variability,
        )

    return realised_terminal, realised_cargo


def _extract_peak_gate_queues(result, gates):
    """Extract peak physical queue stock from the SAME queue_stock output as dashboard."""
    queue_stock = result.get("queue_stock", pd.DataFrame())

    peaks = {g: 0.0 for g in gates}

    if queue_stock is None or queue_stock.empty:
        return peaks

    required = {"gate", "queue_stock"}
    if not required.issubset(queue_stock.columns):
        return peaks

    for gate in gates:
        values = pd.to_numeric(
            queue_stock.loc[queue_stock["gate"] == gate, "queue_stock"],
            errors="coerce",
        ).dropna()

        if not values.empty:
            peaks[gate] = float(values.max())

    return peaks


def _extract_peak_spillback(result):
    """Calculate spillback excess from the model's actual queue_stock and graph.

    This is KPI extraction only. The traffic simulation itself is still entirely
    handled by main.py.
    """
    queue_stock = result.get("queue_stock", pd.DataFrame())
    graph = result.get("graph")

    if queue_stock is None or queue_stock.empty or graph is None:
        return 0.0, {}

    # Same exit-gate approaches used by main.py.
    approaches = {
        "G8": "N06",
        "G9": "N02",
        "G1": "N01",
    }

    total_peak = 0.0
    by_gate = {}

    for gate, upstream in approaches.items():
        if not graph.has_edge(upstream, gate):
            candidates = [
                n for n in graph.nodes
                if str(n).replace("0", "") == upstream.replace("0", "")
            ]
            upstream_node = candidates[0] if candidates else upstream
        else:
            upstream_node = upstream

        if not graph.has_edge(upstream_node, gate):
            by_gate[gate] = 0.0
            continue

        data = graph[upstream_node][gate]
        storage = max(
            1.0,
            float(data.get("capacity_vph", 0.0))
            * float(data.get("travel_time_min", 0.0))
            / 60.0,
        )

        q = queue_stock[
            (queue_stock["gate"] == gate)
            & (queue_stock["operation"] == "EXIT")
        ]

        peak_q = (
            float(pd.to_numeric(q["queue_stock"], errors="coerce").max())
            if not q.empty
            else 0.0
        )

        excess = max(0.0, peak_q - storage)
        by_gate[gate] = excess
        total_peak = max(total_peak, excess)

    return total_peak, by_gate


def _extract_metrics(result, gates):
    """Extract KPIs from the actual run_simulation result."""
    trucks = result.get("trucks", pd.DataFrame())
    gates_df = result.get("gates", pd.DataFrame())
    roads = result.get("roads", pd.DataFrame())
    junctions = result.get("junction_summary", pd.DataFrame())

    peak_queue = _extract_peak_gate_queues(result, gates)
    peak_spillback, spillback_by_gate = _extract_peak_spillback(result)

    peak_network_vc = (
        float(pd.to_numeric(roads["peak_vc"], errors="coerce").max())
        if not roads.empty and "peak_vc" in roads.columns
        else 0.0
    )

    avg_turnaround = (
        float(pd.to_numeric(trucks["total_time_min"], errors="coerce").mean())
        if not trucks.empty and "total_time_min" in trucks.columns
        else 0.0
    )

    avg_gate_wait = (
        float(pd.to_numeric(gates_df["wait_min"], errors="coerce").mean())
        if not gates_df.empty and "wait_min" in gates_df.columns
        else 0.0
    )

    peak_gate_wait = (
        float(pd.to_numeric(gates_df["wait_min"], errors="coerce").max())
        if not gates_df.empty and "wait_min" in gates_df.columns
        else 0.0
    )

    peak_junction_vc = (
        float(pd.to_numeric(junctions["peak_vc"], errors="coerce").max())
        if not junctions.empty and "peak_vc" in junctions.columns
        else 0.0
    )

    row = {
        "peak_queue_all": max(peak_queue.values()) if peak_queue else 0.0,
        "peak_network_vc": peak_network_vc,
        "peak_spillback_trucks": peak_spillback,
        "avg_turn_proxy_min": avg_turnaround,
        "avg_turnaround_min": avg_turnaround,
        "avg_gate_wait_min": avg_gate_wait,
        "peak_gate_wait_min": peak_gate_wait,
        "peak_junction_vc": peak_junction_vc,
    }

    for gate in gates:
        row[f"peak_queue_{gate}"] = peak_queue.get(gate, 0.0)
        row[f"peak_spillback_{gate}"] = spillback_by_gate.get(gate, 0.0)

    return row


def _percentile_summary(df, columns):
    metrics = {}

    for col in columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            metrics[col] = {p: 0.0 for p in PERCENTILES}
        else:
            metrics[col] = {
                p: float(values.quantile(q))
                for p, q in PERCENTILES.items()
            }

    return metrics


def _representative_scenario(df):
    """Return the real simulation closest to the joint median of core KPIs."""
    if df.empty:
        return {}

    target = np.array(
        [
            float(df["peak_queue_all"].median()),
            float(df["peak_network_vc"].median()),
            float(df["avg_turn_proxy_min"].median()),
        ],
        dtype=float,
    )

    x = df[
        ["peak_queue_all", "peak_network_vc", "avg_turn_proxy_min"]
    ].to_numpy(dtype=float)

    scale = np.maximum(np.abs(target), 1.0)
    distance = (((x - target) / scale) ** 2).sum(axis=1)
    idx = int(np.argmin(distance))

    return df.iloc[idx].to_dict()


def run_monte_carlo(
    model,
    simulations=100,
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
    """Run Monte Carlo realisations through the full JIP simulation engine.

    Important modelling convention:
    - Daily scenario demand is FIXED at trucks_per_day.
    - Appointment management is exactly the setting selected by the user.
    - Gate rules, lanes, gate processing times, network capacity and junction
      rules are passed unchanged to main.py.
    - Variability is introduced only through arrival timing, route
      randomisation (if requested), and small terminal/cargo allocation
      perturbations.
    - No separate analytical queue or V/C model is used.
    """

    simulations = int(simulations)
    if simulations < 1:
        raise ValueError("simulations must be at least 1")

    rng = np.random.default_rng(int(seed))

    # Freeze the user-defined scenario inputs. Each realisation gets a copy.
    base_entry_rules = deepcopy(entry_gate_rules or model.ENTRY_GATE_RULES)
    base_exit_rules = deepcopy(exit_gate_rules or model.EXIT_GATE_RULES)
    base_access_rules = deepcopy(
        entry_access_rules or model.ENTRY_ACCESS_RULES
    )
    base_gate_lanes = deepcopy(gate_lanes or model.GATE_LANES)
    base_gate_times = deepcopy(gate_times_sec or model.GATE_TIMES_SEC)
    base_route_overrides = deepcopy(route_overrides or {})
    base_logistics_destination = deepcopy(
        logistics_destination_daily or model.LOGISTICS_DESTINATION_DAILY
    )
    base_junctions = deepcopy(model.JUNCTIONS)

    # These are the gate names actually used by the dashboard.
    gates = list(base_gate_lanes.keys())

    records = []

    for sim in range(simulations):
        # Every realisation gets its own deterministic seed.
        # If ALL variability is zero, use the same seed for every realisation.
        all_variability_zero = (
            float(arrival_variability) == 0.0
            and float(demand_variability) == 0.0
            and (
                route_randomization is None
                or float(route_randomization) == 0.0
            )
        )

        if all_variability_zero:
            simulation_seed = int(seed)
        else:
            simulation_seed = int(rng.integers(0, 2_147_483_647))

        # Keep daily demand fixed. demand_variability is used only to perturb
        # the allocation shares around the scenario definition.
        realised_terminal_shares, realised_cargo_shares = (
            _build_realisation_shares(
                model,
                rng,
                float(demand_variability),
                terminal_shares,
                cargo_shares,
            )
        )

        # Capture stdout because main.py prints a complete simulation summary
        # for every run. The dashboard should remain clean.
        with contextlib.redirect_stdout(io.StringIO()):
            result = model.run_simulation(
                trucks_per_day=int(trucks_per_day),
                route_randomization=(
                    float(route_randomization)
                    if route_randomization is not None
                    else float(model.ROUTE_RANDOMIZATION)
                ),
                seed=simulation_seed,
                entry_gate_rules=deepcopy(base_entry_rules),
                exit_gate_rules=deepcopy(base_exit_rules),
                entry_access_rules=deepcopy(base_access_rules),
                gate_lanes=deepcopy(base_gate_lanes),
                average_speed_kmh=float(model.AVERAGE_SPEED_KMH),
                lanes_per_direction=(
                    int(lanes_per_direction)
                    if lanes_per_direction is not None
                    else int(model.LANES_PER_DIRECTION)
                ),
                capacity_per_lane_vph=(
                    int(capacity_per_lane_vph)
                    if capacity_per_lane_vph is not None
                    else int(model.CAPACITY_PER_LANE_VPH)
                ),
                junctions=deepcopy(base_junctions),
                appointment_management_enabled=bool(management_enabled),
                max_entries_per_hour=int(max_entries_per_hour),
                route_overrides=deepcopy(base_route_overrides),
                logistics_destination_daily=deepcopy(base_logistics_destination),
                gate_times_sec=deepcopy(base_gate_times),
                n2_n1_corridor_lanes=(
                    int(n2_n1_corridor_lanes)
                    if n2_n1_corridor_lanes is not None
                    else int(model.N2_N1_CORRIDOR_LANES)
                ),
                terminal_shares=realised_terminal_shares,
                cargo_shares=realised_cargo_shares,
                arrival_time_variability=float(arrival_variability),
                save_outputs=False,
            )

        metrics = _extract_metrics(result, gates)

        metrics.update(
            {
                "simulation": sim + 1,
                "daily_demand": int(trucks_per_day),
                "seed": simulation_seed,
                "appointment_management": bool(management_enabled),
                "max_entries_per_hour": int(max_entries_per_hour),
            }
        )

        records.append(metrics)

    df = pd.DataFrame(records)

    metric_columns = [
        "peak_queue_all",
        "peak_network_vc",
        "peak_spillback_trucks",
        "peak_marshalling_queue",
        "avg_turn_proxy_min",
        "avg_turnaround_min",
        "avg_gate_wait_min",
        "peak_gate_wait_min",
        "peak_junction_vc",
    ]

    metric_columns += [f"peak_queue_{g}" for g in gates]
    metric_columns += [f"peak_spillback_{g}" for g in gates]

    metric_columns = [c for c in metric_columns if c in df.columns]
    metrics = _percentile_summary(df, metric_columns)

    representative = _representative_scenario(df)

    # Use the last actual model result only for static metadata. No analytical
    # capacity calculations are performed here.
    last_result = result
    graph = last_result.get("graph")

    network_capacity = {}
    if graph is not None:
        for a, b, data in graph.edges(data=True):
            if "capacity_vph" in data:
                network_capacity[f"{a} → {b}"] = float(data["capacity_vph"])

    # Gate capacities are taken from the actual GateManager configuration
    # through the same gate times/lane assumptions used by main.py. We do not
    # use these to calculate queues; they are metadata only.
    gate_capacities = {}
    for gate, lanes_cfg in base_gate_lanes.items():
        for operation in ("entry", "exit"):
            key = f"{gate} {operation.upper()}"
            lanes = int(lanes_cfg.get(operation, 0))
            gate_capacities[key] = {
                "lanes": lanes,
                "configured": True,
            }

    return {
        "results": df,
        "metrics": metrics,
        "representative": representative,
        "gate_capacities_vph": gate_capacities,
        "gate_probabilities": {},
        "network_capacity_vph": network_capacity,
        "network_links": list(network_capacity.keys()),
    }
