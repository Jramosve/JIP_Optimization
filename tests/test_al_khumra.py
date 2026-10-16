"""
Al-Khumra logistics corridor checks (main.py and monte_carlo.py).

The Al-Khumra flow is a subset of the daily demand: it only replaces the
entry/exit gates of the tagged trucks and never adds or removes trucks.
"""

import contextlib
import io

import numpy as np
import pytest

import main as model
import monte_carlo as mcm

LANES_G6 = {**model.GATE_LANES, "G6": {"entry": 2, "exit": 0}}
FLOWS = {
    "RSGT": {"mode": "absolute", "value": 2000, "entry_gate": "G6", "exit_gate": "G8"},
    "DPW": {"mode": "absolute", "value": 1500, "entry_gate": "G9", "exit_gate": "G8"},
    "MPT": {"mode": "share", "value": 0.125, "entry_gate": "G9", "exit_gate": "G8"},
    "LOGISTICS": {"mode": "share", "value": 0.0, "entry_gate": "G9", "exit_gate": "G8"},
}
ZERO_FLOWS = {t: {**c, "value": 0, "entry_gate": "G1", "exit_gate": "G1"} for t, c in FLOWS.items()}
COMPARE = ["truck_id", "arrival_min", "terminal", "cargo", "entry_gate", "exit_gate", "network_destination"]


def run_main(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return model.run_simulation(save_outputs=False, **kwargs)


@pytest.fixture(scope="module")
def runs():
    kw = dict(trucks_per_day=20000, seed=7, arrival_time_variability=0.15, gate_lanes=LANES_G6)
    return {
        "none": run_main(**kw),
        "zero": run_main(**kw, al_khumra_flows=ZERO_FLOWS),
        "ak": run_main(**kw, al_khumra_flows=FLOWS),
    }


def test_zero_al_khumra_reproduces_model_exactly(runs):
    a, b = runs["none"], runs["zero"]
    assert a["trucks"].equals(b["trucks"])
    assert a["queue_stock"].equals(b["queue_stock"])
    assert a["gates"].equals(b["gates"])
    assert (a["trucks"]["origin"] == model.GENERAL_ORIGIN).all()


def test_al_khumra_is_a_subset_of_demand(runs):
    t = runs["ak"]["trucks"]
    assert len(t) == 20000
    ak = t[t["origin"] == model.AL_KHUMRA_ORIGIN]
    assert len(ak) == 2000 + 1500 + 500  # MPT 12.5% of 4,000
    assert t["terminal"].value_counts().equals(runs["none"]["trucks"]["terminal"].value_counts())


def test_al_khumra_trucks_use_configured_gates(runs):
    t = runs["ak"]["trucks"]
    for terminal, cfg in FLOWS.items():
        sub = t[(t["origin"] == model.AL_KHUMRA_ORIGIN) & (t["terminal"] == terminal)]
        if len(sub):
            assert set(sub["entry_gate"]) == {cfg["entry_gate"]}
            assert set(sub["exit_gate"]) == {cfg["exit_gate"]}
    assert set(runs["ak"]["gates"].query("operation == 'ENTRY' and gate == 'G6'")["truck_id"]) == set(
        t.query("origin == 'AL_KHUMRA' and terminal == 'RSGT'")["truck_id"])


def test_general_trucks_are_unchanged(runs):
    t, base = runs["ak"]["trucks"], runs["none"]["trucks"]
    general = t[t["origin"] == model.GENERAL_ORIGIN][COMPARE].set_index("truck_id")
    assert general.equals(base[COMPARE].set_index("truck_id").loc[general.index])
    keep = ["arrival_min", "terminal", "cargo", "network_destination"]
    assert t.set_index("truck_id")[keep].equals(base.set_index("truck_id")[keep])


def test_larger_value_extends_the_selected_set():
    def selected(value):
        flows = {**ZERO_FLOWS, "DPW": {**FLOWS["DPW"], "value": value}}
        t = run_main(trucks_per_day=10500, seed=3, al_khumra_flows=flows)["trucks"]
        return set(t.query("origin == 'AL_KHUMRA'")["truck_id"])
    small, large = selected(500), selected(900)
    assert len(small) == 500 and len(large) == 900 and small <= large


@pytest.mark.parametrize("flows,lanes,message", [
    ({**FLOWS, "DPW": {**FLOWS["DPW"], "value": 99999}}, LANES_G6, "only has"),
    (FLOWS, model.GATE_LANES, "0 entry lanes"),
    ({**FLOWS, "MPT": {**FLOWS["MPT"], "value": 1.5}}, LANES_G6, "between 0 and 1"),
    ({**FLOWS, "DPW": {**FLOWS["DPW"], "value": 10.5}}, LANES_G6, "whole number"),
    ({**FLOWS, "MPT": {**FLOWS["MPT"], "mode": "pct"}}, LANES_G6, "mode"),
])
def test_invalid_al_khumra_inputs_are_rejected(flows, lanes, message):
    with pytest.raises(ValueError, match=message):
        run_main(trucks_per_day=20000, seed=7, gate_lanes=lanes, al_khumra_flows=flows)


def test_monte_carlo_zero_al_khumra_is_identical():
    kw = dict(simulations=60, trucks_per_day=10500, arrival_variability=0.15, seed=11)
    a = mcm.run_monte_carlo(model, **kw)["results"]
    b = mcm.run_monte_carlo(model, **kw, al_khumra_flows=ZERO_FLOWS)["results"]
    assert a.equals(b)
    assert (a["al_khumra_trucks"] == 0).all()


def test_monte_carlo_al_khumra_volume_and_gates():
    res = mcm.run_monte_carlo(
        model, 60, trucks_per_day=20000, seed=7, gate_lanes=LANES_G6, al_khumra_flows=FLOWS
    )["results"]
    assert (res["al_khumra_trucks"] == 4000).all()
    assert (res["peak_queue_G6_ENTRY"] > 0).all()


def test_replay_main_al_khumra_trucks_through_gate_engine(runs, monkeypatch):
    """main.py's Al-Khumra truck list replayed through the fast gate engine
    must give identical entry-gate queues (same check as the base replay)."""
    main_run = runs["ak"]
    trucks = main_run["trucks"]

    def replay_batch(cfg, rng, batch):
        gate_idx = {g: i for i, g in enumerate(cfg["gates"])}
        cargo = (trucks["cargo"] == "EMPTY").astype(int).to_numpy()[None]
        entry = trucks["entry_gate"].map(gate_idx).to_numpy()[None]
        _, entry_kpis = mcm._gate_stage(
            entry, trucks["arrival_min"].to_numpy()[None], cfg["service_min"][0][cargo],
            cfg["gates"], cfg["entry_lanes"], cfg["snapshot_times"], presorted=True,
        )
        zeros = np.zeros(1)
        marshalling = {k: zeros for k in (
            "peak_marshalling_queue", "avg_marshalling_wait", "peak_marshalling_wait",
            "marshalling_truck_hours", "last_entry_hour",
        )}
        return entry_kpis, {}, marshalling

    monkeypatch.setattr(mcm, "_simulate_batch", replay_batch)
    row = mcm.run_monte_carlo(
        model, 1, 20000, 0.15, 0.0, True, 500, 7, gate_lanes=LANES_G6, al_khumra_flows=FLOWS
    )["results"].iloc[0]
    qs = main_run["queue_stock"]
    for gate, grp in qs[qs["operation"] == "ENTRY"].groupby("gate"):
        assert row[f"peak_queue_{gate}_ENTRY"] == grp["queue_stock"].max()


# ---- Separate entry / exit values ---------------------------------------

SPLIT_FLOWS = {
    **ZERO_FLOWS,
    "RSGT": {"mode": "share", "entry_value": 0.20, "exit_value": 0.30, "entry_gate": "G6", "exit_gate": "G8"},
    "DPW": {"mode": "absolute", "entry_value": 1200, "exit_value": 400, "entry_gate": "G9", "exit_gate": "G1"},
}


@pytest.fixture(scope="module")
def split_run():
    return run_main(trucks_per_day=20000, seed=7, arrival_time_variability=0.15,
                    gate_lanes=LANES_G6, al_khumra_flows=SPLIT_FLOWS)


def test_entry_and_exit_volumes_are_separate(split_run):
    t = split_run["trucks"]
    rsgt, dpw = t[t.terminal == "RSGT"], t[t.terminal == "DPW"]
    assert rsgt.al_khumra_entry.sum() == 1600 and rsgt.al_khumra_exit.sum() == 2400  # 20% / 30% of 8,000
    assert dpw.al_khumra_entry.sum() == 1200 and dpw.al_khumra_exit.sum() == 400
    assert len(t) == 20000


def test_entry_and_exit_selections_are_nested(split_run):
    t = split_run["trucks"].set_index("truck_id")
    for terminal in ("RSGT", "DPW"):
        sub = t[t.terminal == terminal]
        entering, leaving = set(sub.index[sub.al_khumra_entry]), set(sub.index[sub.al_khumra_exit])
        small, large = sorted([entering, leaving], key=len)
        assert small <= large


def test_each_direction_uses_its_own_gate(split_run, runs):
    t = split_run["trucks"].set_index("truck_id")
    base = runs["none"]["trucks"].set_index("truck_id")
    for terminal, cfg in (("RSGT", SPLIT_FLOWS["RSGT"]), ("DPW", SPLIT_FLOWS["DPW"])):
        sub = t[t.terminal == terminal]
        assert set(sub[sub.al_khumra_entry].entry_gate) == {cfg["entry_gate"]}
        assert set(sub[sub.al_khumra_exit].exit_gate) == {cfg["exit_gate"]}
        # The other direction keeps the general gate rules.
        no_entry = sub.index[~sub.al_khumra_entry]
        no_exit = sub.index[~sub.al_khumra_exit]
        assert t.loc[no_entry, "entry_gate"].equals(base.loc[no_entry, "entry_gate"])
        assert t.loc[no_exit, "exit_gate"].equals(base.loc[no_exit, "exit_gate"])
    others = t.index[~t.terminal.isin(["RSGT", "DPW"])]
    assert t.loc[others, ["entry_gate", "exit_gate"]].equals(base.loc[others, ["entry_gate", "exit_gate"]])


def test_equal_entry_and_exit_values_match_single_value(runs):
    same = {t: {k: v for k, v in c.items() if k != "value"} | {"entry_value": c["value"], "exit_value": c["value"]}
            for t, c in FLOWS.items()}
    r = run_main(trucks_per_day=20000, seed=7, arrival_time_variability=0.15, gate_lanes=LANES_G6, al_khumra_flows=same)
    assert r["trucks"].equals(runs["ak"]["trucks"])
    assert r["queue_stock"].equals(runs["ak"]["queue_stock"])


def test_exit_only_flow_does_not_need_entry_lanes():
    # G6 has 0 entry lanes by default; an exit-only flow must not require them.
    flows = {**ZERO_FLOWS, "MPT": {"mode": "share", "entry_value": 0.0, "exit_value": 0.5, "entry_gate": "G6", "exit_gate": "G9"}}
    t = run_main(trucks_per_day=10500, seed=3, al_khumra_flows=flows)["trucks"]
    assert t[t.terminal == "MPT"].al_khumra_exit.sum() == 1050
    assert not t.al_khumra_entry.any()
    with pytest.raises(ValueError, match="0 entry lanes"):
        run_main(trucks_per_day=10500, seed=3,
                 al_khumra_flows={**flows, "MPT": {**flows["MPT"], "entry_value": 0.1}})


def test_monte_carlo_entry_and_exit_volumes(split_run):
    res = mcm.run_monte_carlo(
        model, 40, trucks_per_day=20000, arrival_variability=0.15, seed=7,
        gate_lanes=LANES_G6, al_khumra_flows=SPLIT_FLOWS,
    )["results"]
    t = split_run["trucks"]
    assert (res["al_khumra_entry_trucks"] == t.al_khumra_entry.sum()).all()
    assert (res["al_khumra_exit_trucks"] == t.al_khumra_exit.sum()).all()
    assert (res["al_khumra_trucks"] == (t.al_khumra_entry | t.al_khumra_exit).sum()).all()
    assert (res["peak_queue_G6_ENTRY"] > 0).all()
