"""
Fast gate-congestion Monte Carlo model for Jeddah Islamic Port.

This model deliberately does NOT run the full SimPy road/network model.
It reproduces, from main.py:

- daily demand, terminal split (largest-remainder) and shuffle
- observed 24-hour profile, hourly variability and within-hour spacing
- ELM appointment cap with FIFO backlog
- cargo, entry-gate, exit-gate and MPT destination choices
- gate lanes and gate service times (FCFS multi-lane, as simpy.Resource)
- the full truck cycle that feeds the exit gates:
      entry gate finish + entry route + terminal process + internal
      crossing + exit route
  Route times use the same shortest free-flow paths (or route overrides)
  plus junction service time and manoeuvre delay.
- the same KPIs as the dashboard: queue stock (waiting + in service)
  sampled every 5 minutes until 24:00, and gate queue time.

Not modelled: road-link and junction queueing and exit-approach spillback.
These only matter when the internal network itself is saturated; use
main.py to validate those cases.

All simulations of a batch are advanced together with numpy, so the cost
per truck is shared by every scenario in the batch.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


PERCENTILES = {
    "P10": 0.10,
    "P50": 0.50,
    "P90": 0.90,
    "P95": 0.95,
    "P99": 0.99,
}

QUEUE_SNAPSHOT_INTERVAL_MIN = 5.0

# Exact base hourly profile used by main.py.
BASE_HOURLY_SHARES = np.array([
    0.035, 0.030, 0.021, 0.015, 0.018, 0.019,
    0.036, 0.036, 0.043, 0.048, 0.048, 0.051,
    0.050, 0.048, 0.055, 0.056, 0.053, 0.056,
    0.048, 0.046, 0.047, 0.051, 0.048, 0.042,
], dtype=float)
BASE_HOURLY_SHARES /= BASE_HOURLY_SHARES.sum()

# Upper bound on batch_size * trucks_per_day. Small batches keep the
# working arrays in cache, which is faster than one large batch.
MAX_BATCH_CELLS = 1_000_000

OPERATIONS = ("ENTRY", "EXIT")

_GRAPH_CACHE = {}


# ============================================================
# Configuration helpers
# ============================================================

def _normalise(values):
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = values.sum()
    if total <= 0:
        raise ValueError("Shares must sum to a positive value.")
    return values / total


def _perturb(base, rng, variability, size):
    """
    Return (size, len(base)) share matrices. Each row is a lognormal
    perturbation of the base shares, renormalised to 1.
    """
    base = _normalise(base)
    if variability <= 0:
        return np.broadcast_to(base, (size, len(base))).copy()
    raw = base * np.exp(rng.normal(0.0, variability, (size, len(base))))
    return raw / raw.sum(axis=1, keepdims=True)


def _largest_remainder(total, shares):
    """
    Row-wise integer allocation of `total` by `shares` (same principle
    as main.allocate_integer_counts).
    """
    raw = shares * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total) - counts.sum(axis=1)
    order = np.argsort(-(raw - counts), axis=1, kind="stable")
    rank = np.argsort(order, axis=1, kind="stable")
    counts += rank < remainder[:, None]
    return counts


def _categorical(u, cumulative, group):
    """
    Vectorised categorical draw: `cumulative` is (groups, k) and `group`
    selects the row used for each uniform draw in `u`.
    """
    k = cumulative.shape[1]
    idx = np.zeros(u.shape, dtype=np.int64)
    for j in range(k - 1):
        idx += u >= cumulative[group, j]
    return idx


def _load_graph(model):
    key = id(model)
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = model.build_network(model.read_kml_routes())
    return _GRAPH_CACHE[key]


def _route_time_min(
    model,
    graph,
    start,
    end,
    speed_kmh,
    junctions,
    route_overrides,
):
    """
    Uncongested route time used by main.simulate_truck: link travel at
    the average speed plus junction service (60 / capacity) and
    manoeuvre delay at every intermediate junction.
    """
    path = None
    if route_overrides:
        path = route_overrides.get((start, end))
    if not path:
        path = nx.shortest_path(
            graph,
            start,
            end,
            weight="travel_time_min",
        )

    minutes = 0.0
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        data = graph[a][b]
        if "distance_km" in data and speed_kmh:
            minutes += float(data["distance_km"]) / float(speed_kmh) * 60.0
        else:
            minutes += float(data["travel_time_min"])
        if i > 0 and a in junctions:
            cfg = junctions[a]
            minutes += 60.0 / float(cfg["capacity_vph"])
            minutes += float(cfg.get("maneuver_delay_sec", 0.0)) / 60.0
    return minutes


# ============================================================
# Demand generation (vectorised over scenarios)
# ============================================================

def _arrival_times(rng, batch, trucks, variability, cap):
    """
    (batch, trucks) sorted arrival times, reproducing
    main.generate_arrival_times and main.apply_entry_appointment_cap.
    """
    hourly = np.broadcast_to(BASE_HOURLY_SHARES, (batch, 24)).copy()
    if variability > 0:
        hourly *= np.exp(rng.normal(0.0, variability * 0.12, (batch, 24)))
        hourly /= hourly.sum(axis=1, keepdims=True)
    counts = _largest_remainder(trucks, hourly)

    if cap is not None:
        # FIFO backlog: released appointments are spaced uniformly
        # through each hour, and can run past 24:00.
        released = []
        backlog = np.zeros(batch, dtype=np.int64)
        hour = 0
        while hour < 24 or backlog.any():
            demand = backlog + (counts[:, hour] if hour < 24 else 0)
            release = np.minimum(demand, cap)
            released.append(release)
            backlog = demand - release
            hour += 1
        counts = np.stack(released, axis=1)
        jitter_scale = 0.0
    else:
        jitter_scale = 0.25

    hours = counts.shape[1]
    flat_counts = counts.ravel()
    hour_of = np.repeat(np.tile(np.arange(hours), batch), flat_counts)
    count_of = np.repeat(flat_counts, flat_counts)
    starts = np.cumsum(flat_counts) - flat_counts
    position = np.arange(flat_counts.sum()) - np.repeat(starts, flat_counts)

    frac = (position + 0.5) / count_of
    if jitter_scale:
        frac += rng.uniform(-jitter_scale, jitter_scale, frac.shape) / count_of
    minute = np.clip(frac * 60.0, 0.0, 59.999)
    times = (hour_of * 60.0 + minute).reshape(batch, trucks)
    # Already sorted except, at most, for clipped boundary values.
    times.sort(axis=1)
    return times


def _terminal_assignments(rng, batch, trucks, terminal_shares):
    counts = _largest_remainder(trucks, terminal_shares)
    codes = np.empty((batch, trucks), dtype=np.int64)
    for s in range(batch):
        row = np.repeat(np.arange(counts.shape[1]), counts[s])
        codes[s] = rng.permutation(row)
    return codes


# ============================================================
# Gate queue engine
# ============================================================

def _fcfs_start_times(arrivals, service_min, lanes):
    """
    FCFS multi-lane gate (identical to a simpy.Resource with `lanes`
    servers) for a whole batch at once.

    arrivals: (batch, n) sorted per row, padded with +inf.
    service_min: (batch, n).
    Returns service start times (inf for padding).
    """
    batch, n = arrivals.shape
    start = np.empty_like(arrivals)
    if n == 0:
        return start

    rows = np.arange(batch)
    lane_free = np.zeros((batch, int(lanes)))

    for j in range(n):
        if lanes == 1:
            s = np.maximum(arrivals[:, j], lane_free[:, 0])
            lane_free[:, 0] = s + service_min[:, j]
        else:
            k = lane_free.argmin(axis=1)
            s = np.maximum(arrivals[:, j], lane_free[rows, k])
            lane_free[rows, k] = s + service_min[:, j]
        start[:, j] = s
    return start


def _gate_stage(
    gate_codes,
    arrivals,
    service_min,
    gates,
    lanes,
    snapshot_times,
    presorted=False,
):
    """
    Simulate one operation (ENTRY or EXIT) at every gate.

    gate_codes, arrivals, service_min: (batch, trucks). Set `presorted`
    when each row of `arrivals` is already sorted. Returns per-truck
    finish times and per-gate KPIs.
    """
    batch, trucks = arrivals.shape

    # Order every row by (gate, arrival time) so each gate's trucks are
    # a contiguous, FCFS-ordered block.
    if presorted:
        order = np.argsort(gate_codes, axis=1, kind="stable")
    else:
        by_time = np.argsort(arrivals, axis=1)
        by_gate = np.argsort(
            np.take_along_axis(gate_codes, by_time, axis=1),
            axis=1,
            kind="stable",
        )
        order = np.take_along_axis(by_time, by_gate, axis=1)

    arr_sorted = np.take_along_axis(arrivals, order, axis=1)
    svc_sorted = np.take_along_axis(service_min, order, axis=1)
    counts = np.stack(
        [(gate_codes == g).sum(axis=1) for g in range(len(gates))],
        axis=1,
    )
    offsets = np.cumsum(counts, axis=1) - counts

    # Extra last column collects writes from padding cells.
    finish_sorted = np.full((batch, trucks + 1), np.nan)
    kpis = {}

    for g, gate in enumerate(gates):
        count = counts[:, g]
        width = int(count.max())
        if width == 0:
            kpis[gate] = None
            continue
        if lanes[gate] <= 0:
            raise ValueError(
                f"Gate {gate} receives trucks but has no lanes configured."
            )

        position = np.arange(width)[None, :]
        valid = position < count[:, None]
        idx = np.where(valid, offsets[:, g:g + 1] + position, trucks)
        safe_idx = np.minimum(idx, trucks - 1)

        arr = np.where(
            valid, np.take_along_axis(arr_sorted, safe_idx, axis=1), np.inf
        )
        svc = np.where(
            valid, np.take_along_axis(svc_sorted, safe_idx, axis=1), 0.0
        )

        start = _fcfs_start_times(arr, svc, lanes[gate])
        end = start + svc
        np.put_along_axis(finish_sorted, idx, end, axis=1)

        # Queue stock (waiting + in service) at each snapshot:
        # arrived by t minus finished by t.
        end_sorted = np.sort(end, axis=1)
        stock = np.empty((batch, len(snapshot_times)))
        for s in range(batch):
            stock[s] = (
                np.searchsorted(arr[s], snapshot_times, side="right")
                - np.searchsorted(end_sorted[s], snapshot_times, side="right")
            )

        wait = np.where(valid, start - np.where(valid, arr, 0.0), 0.0)
        kpis[gate] = {
            "trucks": count,
            "peak_stock": stock.max(axis=1),
            "stock_profile": stock,
            "wait_sum": wait.sum(axis=1),
            "wait_max": wait.max(axis=1),
        }

    finish = np.empty((batch, trucks))
    np.put_along_axis(finish, order, finish_sorted[:, :trucks], axis=1)
    return finish, kpis


# ============================================================
# One batch of scenarios
# ============================================================

def _simulate_batch(cfg, rng, batch):
    trucks = cfg["trucks_per_day"]
    n_terms = len(cfg["terminals"])

    # ---- Shares (optional Monte Carlo perturbation) ----------
    terminal_shares = _perturb(
        cfg["terminal_shares"], rng, cfg["demand_variability"], batch
    )
    full_share = np.empty((batch, n_terms))
    for t in range(n_terms):
        full_share[:, t] = _perturb(
            cfg["cargo_shares"][t], rng, cfg["demand_variability"], batch
        )[:, 0]

    # ---- Demand ----------------------------------------------
    terminal = _terminal_assignments(rng, batch, trucks, terminal_shares)
    arrival = _arrival_times(
        rng,
        batch,
        trucks,
        cfg["arrival_variability"],
        cfg["max_entries_per_hour"] if cfg["management_enabled"] else None,
    )

    rows = np.arange(batch)[:, None]
    cargo = (
        rng.random((batch, trucks)) >= full_share[rows, terminal]
    ).astype(np.int64)  # 0 = FULL, 1 = EMPTY

    flow = terminal * 2 + cargo
    entry_gate = _categorical(
        rng.random((batch, trucks)), cfg["entry_cum"], flow
    )
    exit_gate = _categorical(
        rng.random((batch, trucks)), cfg["exit_cum"], flow
    )
    destination = _categorical(
        rng.random((batch, trucks)), cfg["dest_cum"], terminal
    )
    # Map the per-terminal destination choice onto global node codes.
    dest_node = cfg["dest_node"][terminal, destination]

    # ---- Entry gates -----------------------------------------
    entry_service = cfg["service_min"][0][cargo]
    entry_finish, entry_kpis = _gate_stage(
        entry_gate,
        arrival,
        entry_service,
        cfg["gates"],
        cfg["entry_lanes"],
        cfg["snapshot_times"],
        presorted=True,
    )

    # ---- Truck cycle to the exit gate ------------------------
    exit_arrival = (
        entry_finish
        + cfg["entry_route_min"][entry_gate, dest_node]
        + cfg["terminal_min"][terminal, cargo]
        + cfg["exit_route_min"][dest_node, exit_gate]
    )
    exit_service = cfg["service_min"][1][cargo]
    _, exit_kpis = _gate_stage(
        exit_gate,
        exit_arrival,
        exit_service,
        cfg["gates"],
        cfg["exit_lanes"],
        cfg["snapshot_times"],
    )

    return entry_kpis, exit_kpis


# ============================================================
# Results
# ============================================================

def _percentile_summary(df, columns):
    metrics = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        metrics[column] = {
            label: float(values.quantile(q)) if not values.empty else 0.0
            for label, q in PERCENTILES.items()
        }
    return metrics


def _batch_rows(cfg, entry_kpis, exit_kpis, batch, first_sim, seed):
    gates = cfg["gates"]
    zeros = np.zeros(batch)
    data = {
        "simulation": np.arange(first_sim, first_sim + batch),
        "seed": np.full(batch, int(seed)),
        "daily_demand": np.full(batch, cfg["trucks_per_day"]),
    }

    total_wait = np.zeros(batch)
    total_trucks = np.zeros(batch)
    peak_wait = np.zeros(batch)
    profile_all = np.zeros((batch, len(cfg["snapshot_times"])))

    for gate in gates:
        gate_peak = zeros.copy()
        gate_wait = zeros.copy()
        gate_wait_max = zeros.copy()
        gate_trucks = zeros.copy()
        for operation, kpis in (("ENTRY", entry_kpis), ("EXIT", exit_kpis)):
            k = kpis.get(gate)
            prefix = f"{gate}_{operation}"
            if k is None:
                data[f"peak_queue_{prefix}"] = zeros
                data[f"avg_wait_{prefix}"] = zeros
                data[f"peak_wait_{prefix}"] = zeros
                continue
            n = k["trucks"]
            avg = np.divide(k["wait_sum"], n, out=zeros.copy(), where=n > 0)
            wmax = np.where(n > 0, k["wait_max"], 0.0)
            data[f"peak_queue_{prefix}"] = k["peak_stock"]
            data[f"avg_wait_{prefix}"] = avg
            data[f"peak_wait_{prefix}"] = wmax

            gate_peak = np.maximum(gate_peak, k["peak_stock"])
            gate_wait += k["wait_sum"]
            gate_trucks += n
            gate_wait_max = np.maximum(gate_wait_max, wmax)
            profile_all = np.maximum(profile_all, k["stock_profile"])

        data[f"peak_queue_{gate}"] = gate_peak
        data[f"peak_wait_{gate}"] = gate_wait_max
        data[f"avg_wait_{gate}"] = np.divide(
            gate_wait, gate_trucks, out=zeros.copy(), where=gate_trucks > 0
        )
        total_wait += gate_wait
        total_trucks += gate_trucks
        peak_wait = np.maximum(peak_wait, gate_wait_max)

    data["peak_queue_all"] = np.max(
        [data[f"peak_queue_{g}"] for g in gates], axis=0
    )
    data["avg_wait_all"] = np.divide(
        total_wait, total_trucks, out=zeros.copy(), where=total_trucks > 0
    )
    data["peak_wait_all"] = peak_wait
    data["peak_queue_hour"] = (
        cfg["snapshot_times"][profile_all.argmax(axis=1)] // 60
    ).astype(int)
    return pd.DataFrame(data)


# ============================================================
# Public API
# ============================================================

def run_monte_carlo(
    model,
    simulations=1000,
    trucks_per_day=None,
    arrival_variability=None,
    demand_variability=0.0,
    management_enabled=None,
    max_entries_per_hour=None,
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
    workers=None,
    average_speed_kmh=None,
    junctions=None,
    terminal_process_min=None,
    graph=None,
    batch_size=None,
    progress_callback=None,
    **kwargs,
):
    """
    Fast gate-congestion Monte Carlo.

    `model` is main.py: it supplies defaults and the road network used
    for route times. No SimPy simulation is executed here.

    Entry gates follow `entry_access_rules` (as main.create_trucks does);
    `entry_gate_rules` is only a fallback when access rules are missing.
    """
    simulations = int(simulations)
    if simulations < 1:
        raise ValueError("simulations must be at least 1.")

    # --------------------------------------------------------
    # Defaults from main.py
    # --------------------------------------------------------
    if trucks_per_day is None:
        trucks_per_day = model.TRUCK_MOVEMENTS_PER_DAY
    if arrival_variability is None:
        arrival_variability = model.ARRIVAL_TIME_RANDOMIZATION
    if management_enabled is None:
        management_enabled = model.APPOINTMENT_MANAGEMENT_ENABLED
    if max_entries_per_hour is None:
        max_entries_per_hour = model.MAX_PORT_ENTRIES_PER_HOUR
    if entry_access_rules is None:
        entry_access_rules = entry_gate_rules or model.ENTRY_ACCESS_RULES
    if exit_gate_rules is None:
        exit_gate_rules = model.EXIT_GATE_RULES
    if gate_lanes is None:
        gate_lanes = model.GATE_LANES
    if gate_times_sec is None:
        gate_times_sec = model.GATE_TIMES_SEC
    if terminal_shares is None:
        terminal_shares = model.TERMINAL_SHARES
    if cargo_shares is None:
        cargo_shares = model.CARGO_SHARES
    if average_speed_kmh is None:
        average_speed_kmh = model.AVERAGE_SPEED_KMH
    if junctions is None:
        junctions = model.JUNCTIONS
    if terminal_process_min is None:
        terminal_process_min = model.TERMINAL_PROCESS_MIN
    if route_overrides is None:
        route_overrides = model.ROUTE_OVERRIDES
    if graph is None:
        graph = _load_graph(model)

    trucks_per_day = int(trucks_per_day)
    arrival_variability = float(arrival_variability)
    demand_variability = float(demand_variability)
    management_enabled = bool(management_enabled)
    max_entries_per_hour = int(max_entries_per_hour)

    if not 0 <= arrival_variability <= 1:
        raise ValueError("arrival_variability must be between 0 and 1.")
    if management_enabled and max_entries_per_hour <= 0:
        raise ValueError("max_entries_per_hour must be > 0.")

    # --------------------------------------------------------
    # Encode configuration as arrays
    # --------------------------------------------------------
    terminals = list(terminal_shares.keys())
    gates = list(gate_lanes.keys())
    for rules in (entry_access_rules, exit_gate_rules):
        for options in rules.values():
            for gate, _ in options:
                if gate not in gates:
                    gates.append(gate)
    gate_index = {g: i for i, g in enumerate(gates)}

    for gate, lanes_cfg in gate_lanes.items():
        for operation in ("entry", "exit"):
            if int(lanes_cfg.get(operation, 0)) < 0:
                raise ValueError(f"Gate {gate} has negative {operation} lanes.")

    def cumulative(options):
        weights = np.zeros(len(gates))
        for gate, share in options:
            weights[gate_index[gate]] += max(float(share), 0.0)
        return np.cumsum(_normalise(weights))

    entry_cum = np.zeros((len(terminals) * 2, len(gates)))
    exit_cum = np.zeros((len(terminals) * 2, len(gates)))
    for t, terminal in enumerate(terminals):
        for c, cargo in enumerate(("FULL", "EMPTY")):
            entry_cum[t * 2 + c] = cumulative(entry_access_rules[(terminal, cargo)])
            exit_cum[t * 2 + c] = cumulative(exit_gate_rules[(terminal, cargo)])

    # Destination nodes per terminal (MPT is split between MPT1-3).
    per_terminal = []
    for terminal in terminals:
        if terminal == "MPT":
            split = model.MPT_DESTINATION_SPLIT
            per_terminal.append((list(split.keys()), list(split.values())))
        else:
            per_terminal.append(([model.get_destination_node(terminal)], [1.0]))
    nodes = sorted({n for names, _ in per_terminal for n in names})
    node_index = {n: i for i, n in enumerate(nodes)}
    width = max(len(names) for names, _ in per_terminal)
    dest_cum = np.ones((len(terminals), width))
    dest_node = np.zeros((len(terminals), width), dtype=np.int64)
    for t, (names, weights) in enumerate(per_terminal):
        dest_cum[t, :len(names)] = np.cumsum(_normalise(weights))
        dest_node[t, :len(names)] = [node_index[n] for n in names]
        dest_node[t, len(names):] = node_index[names[-1]]

    # Route times for every (gate, node) pair that can be used.
    entry_route = np.full((len(gates), len(nodes)), np.nan)
    exit_route = np.full((len(nodes), len(gates)), np.nan)
    overrides = {tuple(k): list(v) for k, v in (route_overrides or {}).items()}
    for t, terminal in enumerate(terminals):
        names = per_terminal[t][0]
        for c, cargo in enumerate(("FULL", "EMPTY")):
            for rules, is_entry in (
                (entry_access_rules, True),
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
