"""
Consistency checks between the full SimPy model (main.py) and the fast
gate Monte Carlo (monte_carlo.py).

Run with:  pytest -q
"""

import contextlib
import io

import numpy as np
import pytest

import main as model
import monte_carlo as mcm


def run_main(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return model.run_simulation(save_outputs=False, **kwargs)


@pytest.fixture(scope="module")
def base_run():
    return run_main(
        trucks_per_day=10500,
        arrival_time_variability=0.15,
        seed=42,
        appointment_management_enabled=True,
        max_entries_per_hour=500,
    )


def test_queue_stock_is_waiting_plus_in_service(base_run):
    qs = base_run["queue_stock"]
    # Waiting trucks must not be counted twice.
    assert (qs["queue_stock"] == qs["queue_waiting"] + qs["in_service"]).all()
    assert (qs["in_service"] <= qs["capacity_lanes"]).all()
    assert (qs["in_service"] == qs["busy"]).mean() > 0.95


def test_marshalling_fields(base_run):
    trucks = base_run["trucks"]
    assert len(trucks) == 10500
    assert (trucks["marshalling_wait_min"] >= 0).all()
    released = np.sort(trucks["arrival_min"].to_numpy())
    # Never more than the ELM cap released in any hour.
    per_hour = np.bincount((released // 60).astype(int))
    assert per_hour.max() <= 500


def test_replay_main_trucks_through_gate_engine(base_run, monkeypatch):
    """Feeding main.py's own truck list to the fast gate engine must
    reproduce its entry-gate queues exactly and exit-gate queues closely."""
    trucks = base_run["trucks"]
    queue_stock = base_run["queue_stock"]
    gates_df = base_run["gates"]

    def replay_batch(cfg, rng, batch):
        gate_idx = {g: i for i, g in enumerate(cfg["gates"])}
        term_idx = {t: i for i, t in enumerate(cfg["terminals"])}
        # Same node order as the engine's route-time tables.
        nodes = sorted({"RSGT", "DPW", "Logipoint", *model.MPT_DESTINATION_SPLIT})
        node_idx = {n: i for i, n in enumerate(nodes)}

        arrival = trucks["arrival_min"].to_numpy()[None]
        cargo = (trucks["cargo"] == "EMPTY").astype(int).to_numpy()[None]
        terminal = trucks["terminal"].map(term_idx).to_numpy()[None]
        entry = trucks["entry_gate"].map(gate_idx).to_numpy()[None]
        exit_ = trucks["exit_gate"].map(gate_idx).to_numpy()[None]
        dest = trucks["network_destination"].map(node_idx).to_numpy()[None]

        entry_finish, entry_kpis = mcm._gate_stage(
            entry, arrival, cfg["service_min"][0][cargo], cfg["gates"],
            cfg["entry_lanes"], cfg["snapshot_times"], presorted=True,
        )
        exit_arrival = (
            entry_finish
            + cfg["entry_route_min"][entry, dest, 0]
            + cfg["terminal_min"][terminal, cargo]
            + cfg["exit_route_min"][dest, exit_, 0]
        )
        _, exit_kpis = mcm._gate_stage(
            exit_, exit_arrival, cfg["service_min"][1][cargo], cfg["gates"],
            cfg["exit_lanes"], cfg["snapshot_times"],
        )
        zeros = np.zeros(1)
        marshalling = {k: zeros for k in (
            "peak_marshalling_queue", "avg_marshalling_wait",
            "peak_marshalling_wait", "marshalling_truck_hours", "last_entry_hour",
        )}
        return entry_kpis, exit_kpis, marshalling

    monkeypatch.setattr(mcm, "_simulate_batch", replay_batch)
    row = mcm.run_monte_carlo(model, 1, 10500, 0.15, 0.0, True, 500, 42)["results"].iloc[0]

    for (gate, operation), grp in queue_stock.groupby(["gate", "operation"]):
        waits = gates_df[(gates_df.gate == gate) & (gates_df.operation == operation)].wait_min
        main_peak = grp["queue_stock"].max()
        replay_peak = row[f"peak_queue_{gate}_{operation}"]
        if operation == "ENTRY":
            assert replay_peak == main_peak
            assert row[f"avg_wait_{gate}_{operation}"] == pytest.approx(waits.mean(), abs=1e-6)
        else:
            # Exit gates differ only by road-link/junction queueing,
            # which the fast model does not simulate.
            assert abs(replay_peak - main_peak) <= 3
            assert row[f"avg_wait_{gate}_{operation}"] == pytest.approx(waits.mean(), abs=0.1)


def test_main_run_inside_monte_carlo_range(base_run):
    results = mcm.run_monte_carlo(model, 300, 10500, 0.15, 0.0, True, 500, 1)["results"]
    qs = base_run["queue_stock"]
    trucks = base_run["trucks"]

    g4 = qs[(qs.gate == "G4") & (qs.operation == "ENTRY")]["queue_stock"].max()
    lo, hi = results["peak_queue_G4_ENTRY"].quantile([0.01, 0.99])
    assert lo <= g4 <= hi

    avg_wait = base_run["gates"]["wait_min"].mean()
    lo, hi = results["avg_wait_all"].quantile([0.01, 0.99])
    assert lo <= avg_wait <= hi

    scheduled = np.sort(trucks["scheduled_arrival_min"].to_numpy())
    released = np.sort(trucks["arrival_min"].to_numpy())
    peak = mcm.marshalling_profile(
        scheduled, released, np.arange(0.0, released[-1] + 60.0, 5.0)
    ).max()
    lo, hi = results["peak_marshalling_queue"].quantile([0.01, 0.99])
    assert lo <= peak <= hi
    assert results["last_entry_hour"].median() == pytest.approx(released[-1] / 60.0, abs=1.0)


def test_monte_carlo_is_reproducible():
    a = mcm.run_monte_carlo(model, 50, 10500, 0.15, 0.03, True, 500, 3)["results"]
    b = mcm.run_monte_carlo(model, 50, 10500, 0.15, 0.03, True, 500, 3)["results"]
    assert a.equals(b)


def test_no_cap_has_no_marshalling_wait():
    results = mcm.run_monte_carlo(model, 50, 10500, 0.15, 0.0, False, 500, 3)["results"]
    assert (results["peak_marshalling_queue"] == 0).all()
    assert (results["marshalling_truck_hours"] == 0).all()


BASE_KW = dict(
    simulations=40, trucks_per_day=10500, arrival_variability=0.15,
    demand_variability=0.0, management_enabled=True, max_entries_per_hour=500, seed=5,
)


def _with_rule(rules, key, options):
    out = {k: list(v) for k, v in rules.items()}
    out[key] = options
    return out


SENSITIVITY_CASES = {
    "trucks_per_day": dict(trucks_per_day=15000),
    "arrival_variability": dict(arrival_variability=0.25),
    "seed": dict(seed=6),
    "management_enabled": dict(management_enabled=False),
    "max_entries_per_hour": dict(max_entries_per_hour=400),
    "average_speed_kmh": dict(average_speed_kmh=10.0),
    "junctions": dict(junctions={k: {**v, "capacity_vph": 60} for k, v in model.JUNCTIONS.items()}),
    "gate_lanes": dict(gate_lanes={**model.GATE_LANES, "G4": {"entry": 4, "exit": 0}}),
    "gate_times_sec": dict(gate_times_sec={**model.GATE_TIMES_SEC, "FULL_ENTRY": 45.0, "FULL_EXIT": 60.0}),
    "terminal_shares": dict(terminal_shares={"RSGT": 0.60, "DPW": 0.20, "MPT": 0.15, "LOGISTICS": 0.05}),
    "cargo_shares": dict(cargo_shares={t: {"FULL": 0.5, "EMPTY": 0.5} for t in model.CARGO_SHARES}),
    "entry_access_rules": dict(entry_access_rules=_with_rule(
        model.ENTRY_ACCESS_RULES, ("RSGT", "FULL"), [("G4", 0.5), ("G9", 0.5)])),
    "exit_gate_rules": dict(exit_gate_rules=_with_rule(
        model.EXIT_GATE_RULES, ("DPW", "FULL"), [("G8", 0.5), ("G1", 0.5)])),
    "terminal_process_min": dict(terminal_process_min={
        t: {"FULL": 30.0, "EMPTY": 30.0} for t in model.TERMINAL_PROCESS_MIN}),
    # A detour G4 > N04 > N05 > N04 > MPT2 instead of the direct G4 > N04 > MPT2.
    "route_overrides": dict(route_overrides={
        ("G4", "MPT2"): ["G4", "N04", "N05", "N04", "MPT2"]}),
}


@pytest.mark.parametrize("name", list(SENSITIVITY_CASES))
def test_every_input_changes_monte_carlo(name):
    """Each dashboard input passed to the Monte Carlo must change its output."""
    kw = SENSITIVITY_CASES[name]
    base = mcm.run_monte_carlo(model, **BASE_KW)["results"]
    changed = mcm.run_monte_carlo(model, **{**BASE_KW, **kw})["results"]
    cols = [c for c in base.columns if c not in ("simulation", "seed")]
    assert not base[cols].equals(changed[cols]), f"{name} has no effect on the Monte Carlo"


def test_route_randomization_matches_main_choice_rule():
    """The JIP KML network has a single path per OD pair today, so route
    randomization only matters once alternatives exist. Check the rule on a
    small network with two routes A > B: direct (10 min) or via C (20 min)."""
    import networkx as nx

    g = nx.DiGraph()
    for a, b, minutes in [("A", "B", 10.0), ("A", "C", 10.0), ("C", "B", 10.0)]:
        g.add_edge(a, b, travel_time_min=minutes)
    times, probs = mcm._route_options(model, g, "A", "B", None, {}, {}, 0.0)
    assert times == [10.0, 20.0] and list(probs) == [1.0, 0.0]

    times, probs = mcm._route_options(model, g, "A", "B", None, {}, {}, 0.5)
    # main.choose_network_path: shortest with prob 1 - r; otherwise weighted
    # by max(0.05, min_cost / cost) = [1, 0.5].
    assert probs == pytest.approx([0.5 + 0.5 * 2 / 3, 0.5 * 1 / 3])
