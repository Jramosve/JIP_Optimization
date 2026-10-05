
"""
Fast gate-only Monte Carlo model for Jeddah Islamic Port.

This model deliberately does NOT run the full road/network SimPy model.
It reproduces the demand, terminal/cargo allocation, gate allocation,
appointment management, gate lanes and gate service times from main.py,
and calculates gate queues only.

Use this for large Monte Carlo experiments (e.g. 10,500 trucks x 1,000
scenarios). Use main.py for full network validation.
"""

from __future__ import annotations

import math
import random
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

QUEUE_SNAPSHOT_INTERVAL_MIN = 5.0

# Exact base hourly profile used by main.py.
BASE_HOURLY_SHARES = np.array([
    0.035, 0.030, 0.021, 0.015, 0.018, 0.019,
    0.036, 0.036, 0.043, 0.048, 0.048, 0.051,
    0.050, 0.048, 0.055, 0.056, 0.053, 0.056,
    0.048, 0.046, 0.047, 0.051, 0.048, 0.042,
], dtype=float)
BASE_HOURLY_SHARES /= BASE_HOURLY_SHARES.sum()


def _normalise_shares(values):
    values = {
        str(k): max(float(v), 0.0)
        for k, v in values.items()
    }
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Shares must sum to a positive value.")
    return {k: v / total for k, v in values.items()}


def _perturb_shares(base, rng, variability):
    base = _normalise_shares(base)
    if float(variability) <= 0:
        return base

    keys = list(base.keys())
    raw = np.array([
        base[k] * np.exp(rng.normal(0.0, float(variability)))
        for k in keys
    ], dtype=float)
    raw /= raw.sum()
    return {k: float(v) for k, v in zip(keys, raw)}


def _build_realisation_shares(
    base_terminal_shares,
    base_cargo_shares,
    rng,
    variability,
):
    terminal = _perturb_shares(
        base_terminal_shares, rng, variability
    )
    cargo = {
        terminal_name: _perturb_shares(
            shares, rng, variability
        )
        for terminal_name, shares in base_cargo_shares.items()
    }
    return terminal, cargo


def _allocate_integer_counts(total, weights):
    """
    Same largest-remainder principle used by main.py.
    """
    keys = list(weights.keys())
    raw = np.array(
        [float(weights[k]) for k in keys],
        dtype=float,
    )
    raw = raw / raw.sum() * int(total)

    base = np.floor(raw).astype(int)
    remainder = int(total) - int(base.sum())

    if remainder > 0:
        order = np.argsort(-(raw - base))
        for idx in order[:remainder]:
            base[idx] += 1

    return {
        k: int(v)
        for k, v in zip(keys, base)
    }


def _weighted_choice(options, weights, rng):
    """
    Python-random equivalent of the weighted selection used in main.py.
    """
    return rng.choices(
        options,
        weights=weights,
        k=1,
    )[0]


def _generate_arrival_times(
    number_of_trucks,
    rng,
    variability,
):
    """
    Reproduce main.py's current arrival-time logic:

    - exact daily demand
    - current 24-hour profile
    - hourly perturbation controlled by variability
    - approximately uniform arrivals within each hour
    - sorted arrival times
    """
    hourly = BASE_HOURLY_SHARES.copy()
    variability = float(variability)

    if not 0 <= variability <= 1:
        raise ValueError(
            "arrival_time_variability must be between 0 and 1."
        )

    if variability > 0:
        noise = np.exp(np.array([
            rng.gauss(0.0, variability * 0.12)
            for _ in range(24)
        ]))
        hourly *= noise
        hourly /= hourly.sum()

    raw = hourly * int(number_of_trucks)
    counts = np.floor(raw).astype(int)
    remainder = int(number_of_trucks) - int(counts.sum())

    if remainder > 0:
        for idx in np.argsort(-(raw - counts))[:remainder]:
            counts[idx] += 1

    arrivals = []

    for hour, count in enumerate(counts):
        if count <= 0:
            continue

        for i in range(int(count)):
            frac = (i + 0.5) / count
            jitter = rng.uniform(
                -0.25 / max(count, 1),
                0.25 / max(count, 1),
            )

            minute = min(
                59.999,
                max(
                    0.0,
                    (frac + jitter) * 60.0,
                ),
            )

            arrivals.append(
                hour * 60.0 + minute
            )

    arrivals.sort()
    return arrivals


def _apply_entry_appointment_cap(
    arrival_times,
    max_entries_per_hour,
):
    """
    Reproduce main.py's appointment logic.

    Demand above the hourly cap becomes FIFO backlog and rolls into
    subsequent hours. Released appointments are spread continuously
    through each hour.

    The function may release trucks after 24:00 if the cap requires it.
    """

    if max_entries_per_hour is None:
        return sorted(arrival_times)

    cap = int(max_entries_per_hour)
    if cap <= 0:
        raise ValueError(
            "max_entries_per_hour must be > 0."
        )

    by_hour = {}

    for t in arrival_times:
        hour = max(
            0,
            int(float(t) // 60),
        )
        by_hour.setdefault(
            hour,
            [],
        ).append(float(t))

    for hour in by_hour:
        by_hour[hour].sort()

    backlog = []
    released_times = []
    hour = 0

    while (
        hour < 24
        or backlog
        or hour in by_hour
    ):
        current = by_hour.get(
            hour,
            [],
        )

        pool = backlog + current

        release = min(
            cap,
            len(pool),
        )

        backlog = pool[release:]

        if release:
            for i in range(release):
                minute = (
                    (i + 0.5)
                    / release
                    * 60.0
                )

                released_times.append(
                    hour * 60.0 + minute
                )

        hour += 1

    if len(released_times) != len(arrival_times):
        raise RuntimeError(
            "Appointment management released "
            f"{len(released_times):,} of "
            f"{len(arrival_times):,} trucks."
        )

    return released_times


def _build_truck_gate_flows(
    trucks_per_day,
    random_seed,
    terminal_shares,
    cargo_shares,
    entry_access_rules,
    exit_gate_rules,
    arrival_variability,
    appointment_management_enabled,
    max_entries_per_hour,
):
    """
    Reproduce the gate-relevant part of main.py's create_trucks().

    Roads, nodes, destinations and routes are deliberately omitted.
    """

    rng = random.Random(
        int(random_seed)
    )

    # Same terminal count allocation as main.py.
    terminal_counts = _allocate_integer_counts(
        trucks_per_day,
        terminal_shares,
    )

    terminal_assignments = []

    for terminal, count in terminal_counts.items():
        terminal_assignments.extend(
            [terminal] * int(count)
        )

    # Same shuffle concept as main.py.
    rng.shuffle(
        terminal_assignments
    )

    # Arrival generation happens before cargo/gate selection in main.py.
    arrival_times = _generate_arrival_times(
        trucks_per_day,
        rng,
        arrival_variability,
    )

    if appointment_management_enabled:
        arrival_times = _apply_entry_appointment_cap(
            arrival_times,
            max_entries_per_hour,
        )

    records = []

    for i in range(
        int(trucks_per_day)
    ):
        terminal = terminal_assignments[i]

        cargo_options = list(
            cargo_shares[terminal].keys()
        )
        cargo_weights = list(
            cargo_shares[terminal].values()
        )

        cargo = _weighted_choice(
            cargo_options,
            cargo_weights,
            rng,
        )

        entry_options = [
            x[0]
            for x in entry_access_rules[
                (terminal, cargo)
            ]
        ]
        entry_weights = [
            x[1]
            for x in entry_access_rules[
                (terminal, cargo)
            ]
        ]

        entry_gate = _weighted_choice(
            entry_options,
            entry_weights,
            rng,
        )

        exit_options = [
            x[0]
            for x in exit_gate_rules[
                (terminal, cargo)
            ]
        ]
        exit_weights = [
            x[1]
            for x in exit_gate_rules[
                (terminal, cargo)
            ]
        ]

        exit_gate = _weighted_choice(
            exit_options,
            exit_weights,
            rng,
        )

        records.append(
            (
                arrival_times[i],
                entry_gate,
                exit_gate,
                cargo,
            )
        )

    return records


def _simulate_gate_queue(
    arrivals,
    service_times_sec,
    lanes,
    snapshot_times,
):
    """
    Deterministic multi-server FCFS gate queue.

    For each truck:
        service_start = max(arrival, earliest lane available)

    Queue stock at a snapshot is the number of trucks that have arrived
    but whose gate service has not started yet.

    This corresponds to the resource.queue concept used by GateManager
    in main.py.
    """
    if lanes <= 0:
        return 0.0

    arrivals = np.asarray(
        arrivals,
        dtype=float,
    )

    if len(arrivals) == 0:
        return 0.0

    service_times = np.asarray(
        service_times_sec,
        dtype=float,
    ) / 60.0

    # Each lane stores its next available time.
    lane_free = np.zeros(
        int(lanes),
        dtype=float,
    )

    start_times = np.empty(
        len(arrivals),
        dtype=float,
    )

    for i, arrival in enumerate(arrivals):
        lane_idx = int(
            np.argmin(lane_free)
        )

        start = max(
            float(arrival),
            float(lane_free[lane_idx]),
        )

        start_times[i] = start

        lane_free[lane_idx] = (
            start
            + service_times[i]
        )

    peak_queue = 0

    # Queue = arrived but not yet started.
    # Since both arrays are sorted, searchsorted gives this directly.
    for t in snapshot_times:
        arrived = int(
            np.searchsorted(
                arrivals,
                t,
                side="right",
            )
        )

        started = int(
            np.searchsorted(
                start_times,
                t,
                side="right",
            )
        )

        queue = max(
            0,
            arrived - started,
        )

        if queue > peak_queue:
            peak_queue = queue

    return float(peak_queue)


def _simulate_one_scenario(
    scenario,
    simulation_number,
    simulation_seed,
):
    """
    Run one gate-only scenario.
    """

    terminal_shares, cargo_shares = (
        _build_realisation_shares(
            scenario["terminal_shares"],
            scenario["cargo_shares"],
            np.random.default_rng(
                int(simulation_seed)
            ),
            scenario["demand_variability"],
        )
    )

    records = _build_truck_gate_flows(
        trucks_per_day=scenario["trucks_per_day"],
        random_seed=simulation_seed,
        terminal_shares=terminal_shares,
        cargo_shares=cargo_shares,
        entry_access_rules=scenario[
            "entry_access_rules"
        ],
        exit_gate_rules=scenario[
            "exit_gate_rules"
        ],
        arrival_variability=scenario[
            "arrival_variability"
        ],
        appointment_management_enabled=scenario[
            "management_enabled"
        ],
        max_entries_per_hour=scenario[
            "max_entries_per_hour"
        ],
    )

    gates = list(
        scenario["gate_lanes"].keys()
    )

    gate_times = scenario[
        "gate_times_sec"
    ]

    # main.py snapshots queues every 5 minutes and stops monitoring
    # at 24:00.
    snapshot_times = np.arange(
        0.0,
        24.0 * 60.0 + 0.0001,
        QUEUE_SNAPSHOT_INTERVAL_MIN,
    )

    peak_queues = {}

    for gate in gates:

        # ------------------------------
        # ENTRY
        # ------------------------------

        entry_lanes = int(
            scenario["gate_lanes"]
            .get(gate, {})
            .get("entry", 0)
        )

        if entry_lanes > 0:

            entry_rows = [
                row
                for row in records
                if row[1] == gate
            ]

            entry_rows.sort(
                key=lambda x: x[0]
            )

            entry_arrivals = [
                row[0]
                for row in entry_rows
            ]

            entry_service = [
                float(
                    gate_times[
                        (
                            "FULL_ENTRY"
                            if row[3] == "FULL"
                            else "EMPTY_ENTRY"
                        )
                    ]
                )
                for row in entry_rows
            ]

            peak_queues[
                f"peak_queue_{gate}"
            ] = _simulate_gate_queue(
                entry_arrivals,
                entry_service,
                entry_lanes,
                snapshot_times,
            )

        else:
            peak_queues[
                f"peak_queue_{gate}"
            ] = 0.0

        # ------------------------------
        # EXIT
        # ------------------------------

        exit_lanes = int(
            scenario["gate_lanes"]
            .get(gate, {})
            .get("exit", 0)
        )

        if exit_lanes > 0:

            exit_rows = [
                row
                for row in records
                if row[2] == gate
            ]

            exit_rows.sort(
                key=lambda x: x[0]
            )

            exit_arrivals = [
                row[0]
                for row in exit_rows
            ]

            exit_service = [
                float(
                    gate_times[
                        (
                            "FULL_EXIT"
                            if row[3] == "FULL"
                            else "EMPTY_EXIT"
                        )
                    ]
                )
                for row in exit_rows
            ]

            exit_peak = _simulate_gate_queue(
                exit_arrivals,
                exit_service,
                exit_lanes,
                snapshot_times,
            )

            # Keep the dashboard's gate-level KPI as the maximum
            # physical queue at the gate across entry and exit.
            peak_queues[
                f"peak_queue_{gate}"
            ] = max(
                peak_queues[
                    f"peak_queue_{gate}"
                ],
                exit_peak,
            )

    row = {
        "simulation": int(
            simulation_number
        ),
        "seed": int(
            simulation_seed
        ),
        "daily_demand": int(
            scenario["trucks_per_day"]
        ),
    }

    for gate in gates:
        row[
            f"peak_queue_{gate}"
        ] = float(
            peak_queues[
                f"peak_queue_{gate}"
            ]
        )

    row["peak_queue_all"] = float(
        max(
            (
                row[
                    f"peak_queue_{gate}"
                ]
                for gate in gates
            ),
            default=0.0,
        )
    )

    return row


def _percentile_summary(
    df,
    columns,
):
    metrics = {}

    for column in columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        metrics[column] = {
            label: (
                float(values.quantile(q))
                if not values.empty
                else 0.0
            )
            for label, q in PERCENTILES.items()
        }

    return metrics


def run_monte_carlo(
    model,
    simulations=1000,
    trucks_per_day=None,
    arrival_variability=None,
    demand_variability=0.03,
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
    **kwargs,
):
    """
    Fast gate-only Monte Carlo.

    The `model` argument is retained for dashboard compatibility.
    main.py is used only as the source of default configuration.
    No full SimPy/network simulation is executed here.
    """

    simulations = int(simulations)

    if simulations < 1:
        raise ValueError(
            "simulations must be at least 1."
        )

    # --------------------------------------------------------
    # Read defaults from main.py
    # --------------------------------------------------------

    if trucks_per_day is None:
        trucks_per_day = int(
            model.TRUCK_MOVEMENTS_PER_DAY
        )

    if arrival_variability is None:
        arrival_variability = float(
            model.ARRIVAL_TIME_RANDOMIZATION
        )

    if management_enabled is None:
        management_enabled = bool(
            model.APPOINTMENT_MANAGEMENT_ENABLED
        )

    if max_entries_per_hour is None:
        max_entries_per_hour = int(
            model.MAX_PORT_ENTRIES_PER_HOUR
        )

    if entry_gate_rules is None:
        entry_gate_rules = model.ENTRY_GATE_RULES

    if exit_gate_rules is None:
        exit_gate_rules = model.EXIT_GATE_RULES

    if entry_access_rules is None:
        entry_access_rules = model.ENTRY_ACCESS_RULES

    if gate_lanes is None:
        gate_lanes = model.GATE_LANES

    if gate_times_sec is None:
        gate_times_sec = model.GATE_TIMES_SEC

    if terminal_shares is None:
        terminal_shares = model.TERMINAL_SHARES

    if cargo_shares is None:
        cargo_shares = model.CARGO_SHARES

    # --------------------------------------------------------
    # Freeze configuration
    # --------------------------------------------------------

    scenario = {
        "trucks_per_day": int(
            trucks_per_day
        ),
        "arrival_variability": float(
            arrival_variability
        ),
        "demand_variability": float(
            demand_variability
        ),
        "management_enabled": bool(
            management_enabled
        ),
        "max_entries_per_hour": int(
            max_entries_per_hour
        ),
        "entry_gate_rules": deepcopy(
            entry_gate_rules
        ),
        "exit_gate_rules": deepcopy(
            exit_gate_rules
        ),
        "entry_access_rules": deepcopy(
            entry_access_rules
        ),
        "gate_lanes": deepcopy(
            gate_lanes
        ),
        "gate_times_sec": deepcopy(
            gate_times_sec
        ),
        "terminal_shares": deepcopy(
            terminal_shares
        ),
        "cargo_shares": deepcopy(
            cargo_shares
        ),
    }

    # --------------------------------------------------------
    # Validate gate rules
    # --------------------------------------------------------

    gates = list(
        scenario["gate_lanes"].keys()
    )

    for gate, cfg in scenario[
        "gate_lanes"
    ].items():

        for operation in (
            "entry",
            "exit",
        ):
            if int(
                cfg.get(
                    operation,
                    0,
                )
            ) < 0:
                raise ValueError(
                    f"Gate {gate} has negative "
                    f"{operation} lanes."
                )

    # --------------------------------------------------------
    # Independent scenario seeds
    # --------------------------------------------------------

    master_rng = np.random.default_rng(
        int(seed)
    )

    seeds = master_rng.integers(
        0,
        2_147_483_647,
        size=simulations,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Run scenarios
    #
    # IMPORTANT:
    # No ProcessPoolExecutor.
    #
    # The simplified model is deliberately lightweight enough
    # that one Streamlit process can run the full experiment
    # without spawning multiple copies of main.py.
    # --------------------------------------------------------

    records = []

    for simulation_number, simulation_seed in enumerate(
        seeds,
        start=1,
    ):

        records.append(
            _simulate_one_scenario(
                scenario=scenario,
                simulation_number=simulation_number,
                simulation_seed=int(
                    simulation_seed
                ),
            )
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    df = pd.DataFrame(
        records
    )

    df = df.sort_values(
        "simulation"
    ).reset_index(
        drop=True
    )

    metric_columns = [
        f"peak_queue_{gate}"
        for gate in gates
    ]

    metric_columns.append(
        "peak_queue_all"
    )

    metrics = _percentile_summary(
        df,
        metric_columns,
    )

    # --------------------------------------------------------
    # Representative scenario
    # --------------------------------------------------------

    representative = {}

    if not df.empty:

        target = float(
            df[
                "peak_queue_all"
            ].median()
        )

        idx = (
            df[
                "peak_queue_all"
            ]
            .sub(target)
            .abs()
            .idxmin()
        )

        representative = (
            df.loc[int(idx)]
            .to_dict()
        )

    # --------------------------------------------------------
    # Gate metadata
    # --------------------------------------------------------

    gate_capacities = {}

    for gate, cfg in scenario[
        "gate_lanes"
    ].items():

        gate_capacities[
            f"{gate} ENTRY"
        ] = {
            "lanes": int(
                cfg.get(
                    "entry",
                    0,
                )
            ),
            "configured": True,
        }

        gate_capacities[
            f"{gate} EXIT"
        ] = {
            "lanes": int(
                cfg.get(
                    "exit",
                    0,
                )
            ),
            "configured": True,
        }

    # --------------------------------------------------------
    # Return dashboard-compatible structure
    # --------------------------------------------------------

    return {
        "results": df,

        "metrics": metrics,

        "representative":
            representative,

        "gate_capacities_vph":
            gate_capacities,

        "network_metrics": {},
        "network_percentiles": {},

        "spillback_metrics": {},
        "spillback_percentiles": {},

        "queue_metrics":
            metrics,

        "queue_percentiles":
            metrics,

        "configuration": {
            "simulations":
                simulations,

            "workers":
                1,

            "model_type":
                "gate_only",

            "trucks_per_day":
                int(trucks_per_day),

            "arrival_variability":
                float(arrival_variability),

            "demand_variability":
                float(demand_variability),

            "management_enabled":
                bool(management_enabled),

            "max_entries_per_hour":
                int(max_entries_per_hour),

            "seed":
                int(seed),

            "roads_simulated":
                False,

            "junctions_simulated":
                False,
        },
    }
