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


def ak_lanes_for(flows, gate_lanes=None):
    """Reserve one lane at every gate and direction used by an active
    Al-Khumra flow (reserved lanes are required for Al-Khumra traffic)."""
    gate_lanes = model.GATE_LANES if gate_lanes is None else gate_lanes
    reserved = {g: {"entry": 0, "exit": 0} for g in gate_lanes}
    for cfg in (flows or {}).values():
        for direction in ("entry", "exit"):
            gate = cfg.get(f"{direction}_gate")
            if model.al_khumra_value(cfg, direction) > 0 and int(gate_lanes.get(gate, {}).get(direction, 0)) > 0:
                reserved[gate][direction] = 1
    return reserved


def _with_ak_lanes(kwargs):
    if "al_khumra_lanes" not in kwargs and kwargs.get("al_khumra_flows"):
        kwargs["al_khumra_lanes"] = ak_lanes_for(kwargs["al_khumra_flows"], kwargs.get("gate_lanes"))
    return kwargs


def run_main(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return model.run_simulation(save_outputs=False, **_with_ak_lanes(kwargs))


def run_mc(*args, **kwargs):
    return mcm.run_monte_carlo(model, *args, **_with_ak_lanes(kwargs))


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
    a = run_mc(**kw)["results"]
    b = run_mc(**kw, al_khumra_flows=ZERO_FLOWS)["results"]
    assert a.equals(b)
    assert (a["al_khumra_trucks"] == 0).all()


def test_monte_carlo_al_khumra_volume_and_gates():
    res = run_mc(
        60, trucks_per_day=20000, seed=7, gate_lanes=LANES_G6, al_khumra_flows=FLOWS
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
        # Al-Khumra entries queue for their gate's reserved lanes.
        entry = (trucks["entry_gate"].map(gate_idx)
                 + len(cfg["gates"]) * trucks["al_khumra_entry"].astype(int)).to_numpy()[None]
        _, entry_kpis = mcm._gate_stage(
            entry, trucks["arrival_min"].to_numpy()[None], cfg["service_min"][0][cargo],
            cfg["stage_gates"], cfg["stage_entry_lanes"], cfg["snapshot_times"], presorted=True,
        )
        zeros = np.zeros(1)
        marshalling = {k: zeros for k in (
            "peak_marshalling_queue", "avg_marshalling_wait", "peak_marshalling_wait",
            "marshalling_truck_hours", "last_entry_hour",
        )}
        return entry_kpis, {}, marshalling

    monkeypatch.setattr(mcm, "_simulate_batch", replay_batch)
    row = run_mc(
        1, 20000, 0.15, 0.0, True, 500, 7, gate_lanes=LANES_G6, al_khumra_flows=FLOWS
    )["results"].iloc[0]
    qs = main_run["queue_stock"]
    for gate, grp in qs[qs["operation"] == "ENTRY"].groupby("gate"):
        assert row[f"peak_queue_{gate}_ENTRY"] == grp["queue_stock"].max()
        if grp["al_khumra_capacity_lanes"].max() > 0:
            assert row[f"peak_queue_{gate}_ENTRY_al_khumra"] == grp["al_khumra_queue_stock"].max()


# ---- Separate entry / exit values ---------------------------------------

SPLIT_FLOWS = {
    **ZERO_FLOWS,
    "RSGT": {"mode": "share", "entry_value": 0.20, "exit_value": 0.30, "entry_gate": "G6", "exit_gate": "G8"},
    "DPW": {"mode": "absolute", "entry_value": 1200, "exit_value": 400, "entry_gate": "G9", "exit_gate": "G9"},
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
    res = run_mc(
        40, trucks_per_day=20000, arrival_variability=0.15, seed=7,
        gate_lanes=LANES_G6, al_khumra_flows=SPLIT_FLOWS,
    )["results"]
    t = split_run["trucks"]
    assert (res["al_khumra_entry_trucks"] == t.al_khumra_entry.sum()).all()
    assert (res["al_khumra_exit_trucks"] == t.al_khumra_exit.sum()).all()
    assert (res["al_khumra_trucks"] == (t.al_khumra_entry | t.al_khumra_exit).sum()).all()
    assert (res["peak_queue_G6_ENTRY"] > 0).all()


# ---- Empty share of Al-Khumra trucks --------------------------------------

EMPTY_FLOWS = {
    **ZERO_FLOWS,
    "RSGT": {"mode": "share", "entry_value": 0.20, "exit_value": 0.30, "entry_gate": "G6",
             "exit_gate": "G8", "empty_share": 0.80},
    "DPW": {"mode": "absolute", "entry_value": 1000, "exit_value": 1000, "entry_gate": "G9",
            "exit_gate": "G8", "empty_share": 0.0},
}


@pytest.fixture(scope="module")
def empty_run():
    return run_main(trucks_per_day=20000, seed=7, arrival_time_variability=0.15,
                    gate_lanes=LANES_G6, al_khumra_flows=EMPTY_FLOWS)


def test_empty_share_of_al_khumra_trucks(empty_run):
    t = empty_run["trucks"]
    rsgt, dpw = t[t.terminal == "RSGT"], t[t.terminal == "DPW"]
    for sub, n, empty in ((rsgt[rsgt.al_khumra_entry], 1600, 1280), (rsgt[rsgt.al_khumra_exit], 2400, 1920)):
        assert len(sub) == n and (sub.cargo == "EMPTY").sum() == empty
    assert (dpw[dpw.al_khumra_entry | dpw.al_khumra_exit].cargo == "FULL").all()
    rsgt_entry, rsgt_exit = set(rsgt.truck_id[rsgt.al_khumra_entry]), set(rsgt.truck_id[rsgt.al_khumra_exit])
    assert rsgt_entry <= rsgt_exit


def test_empty_share_keeps_terminal_mix(empty_run, runs):
    # Cargo is never changed: every truck keeps the cargo of the base run, so
    # the terminal Full/Empty mix is unchanged and other traffic gets fuller.
    t, base = empty_run["trucks"], runs["none"]["trucks"]
    assert t["cargo"].equals(base["cargo"])
    rsgt = t[t.terminal == "RSGT"]
    other = rsgt[~(rsgt.al_khumra_entry | rsgt.al_khumra_exit)]
    assert (other.cargo == "EMPTY").mean() < (rsgt.cargo == "EMPTY").mean()


def test_blank_empty_share_is_unchanged(runs):
    flows = {t: {**c, "empty_share": None} for t, c in FLOWS.items()}
    r = run_main(trucks_per_day=20000, seed=7, arrival_time_variability=0.15, gate_lanes=LANES_G6, al_khumra_flows=flows)
    assert r["trucks"].equals(runs["ak"]["trucks"])


def test_too_many_empty_trucks_is_rejected():
    flows = {**ZERO_FLOWS, "RSGT": {**EMPTY_FLOWS["RSGT"], "exit_value": 0.9, "empty_share": 0.9}}
    with pytest.raises(ValueError, match="EMPTY trucks/day are needed"):
        run_main(trucks_per_day=20000, seed=7, gate_lanes=LANES_G6, al_khumra_flows=flows)
    with pytest.raises(ValueError, match="EMPTY trucks/day are needed"):
        run_mc(5, trucks_per_day=20000, seed=7, gate_lanes=LANES_G6, al_khumra_flows=flows)
    with pytest.raises(ValueError, match="empty share"):
        run_main(trucks_per_day=20000, seed=7, gate_lanes=LANES_G6,
                 al_khumra_flows={**ZERO_FLOWS, "RSGT": {**EMPTY_FLOWS["RSGT"], "empty_share": 1.5}})


def test_monte_carlo_empty_share_matches_main(empty_run):
    res = run_mc(30, trucks_per_day=20000, arrival_variability=0.15, seed=7,
                              gate_lanes=LANES_G6, al_khumra_flows=EMPTY_FLOWS)["results"]
    t = empty_run["trucks"]
    ak = t[t.al_khumra_entry | t.al_khumra_exit]
    assert (res["al_khumra_trucks"] == len(ak)).all()
    assert (res["al_khumra_empty_trucks"] == (ak.cargo == "EMPTY").sum()).all()


# ---- Lanes reserved for Al-Khumra -------------------------------------------

RES_LANES = {**model.GATE_LANES, "G9": {"entry": 3, "exit": 4}}
RES_FLOWS = {
    **ZERO_FLOWS,
    "DPW": {"mode": "share", "entry_value": 0.8, "exit_value": 0.8, "entry_gate": "G9", "exit_gate": "G9"},
}
RESERVED = {**model.AL_KHUMRA_LANES, "G9": {"entry": 1, "exit": 1}}


@pytest.fixture(scope="module")
def reserved_run():
    return run_main(trucks_per_day=15000, seed=3, gate_lanes=RES_LANES,
                    al_khumra_flows=RES_FLOWS, al_khumra_lanes=RESERVED)


def test_reserved_lanes_are_exclusive(reserved_run):
    g = reserved_run["gates"].merge(
        reserved_run["trucks"][["truck_id", "al_khumra_entry", "al_khumra_exit"]], on="truck_id")
    for op, flag in (("ENTRY", "al_khumra_entry"), ("EXIT", "al_khumra_exit")):
        at_g9 = g[(g.gate == "G9") & (g.operation == op)]
        assert (at_g9[at_g9[flag]].lane_group == model.AL_KHUMRA_LANE_GROUP).all()
        assert (at_g9[~at_g9[flag]].lane_group == model.GENERAL_LANE_GROUP).all()
    assert (g[g.gate != "G9"].lane_group == model.GENERAL_LANE_GROUP).all()


def test_gate_queue_is_total_of_both_lane_groups(reserved_run):
    qs = reserved_run["queue_stock"]
    assert not qs.duplicated(["time_min", "gate", "operation"]).any()
    g9 = qs[qs.gate == "G9"].set_index("operation")
    assert g9.loc["EXIT", "capacity_lanes"].max() == 4 and g9.loc["EXIT", "al_khumra_capacity_lanes"].max() == 1
    assert (qs.queue_stock >= qs.al_khumra_queue_stock).all()
    # One Al-Khumra entry lane for 80% of DPW: its own queue builds up while
    # the two general entry lanes stay almost free.
    entry_g9 = qs[(qs.gate == "G9") & (qs.operation == "ENTRY")]
    assert entry_g9.al_khumra_queue_stock.max() > 10 * (entry_g9.queue_stock - entry_g9.al_khumra_queue_stock).max()


def test_reserved_lanes_without_al_khumra_reduce_general_capacity():
    lanes = {**model.AL_KHUMRA_LANES, "G4": {"entry": 1, "exit": 0}}
    base = run_main(trucks_per_day=10500, seed=3)
    r = run_main(trucks_per_day=10500, seed=3, al_khumra_lanes=lanes)
    g4 = lambda x: x["gates"].query("gate == 'G4'").wait_min.mean()
    assert g4(r) > g4(base)


@pytest.mark.parametrize("kwargs,message", [
    (dict(al_khumra_flows=RES_FLOWS, al_khumra_lanes=model.AL_KHUMRA_LANES), "no entry lanes reserved"),
    (dict(al_khumra_lanes={**RESERVED, "G9": {"entry": 4, "exit": 1}}), "only 3 entry lanes"),
    (dict(al_khumra_lanes={**RESERVED, "G9": {"entry": 3, "exit": 1}}), "reserved for Al-Khumra"),
])
def test_invalid_reserved_lanes_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_main(trucks_per_day=15000, seed=3, gate_lanes=RES_LANES, **kwargs)
    with pytest.raises(ValueError, match=message):
        mcm.run_monte_carlo(model, 5, trucks_per_day=15000, seed=3, gate_lanes=RES_LANES, **kwargs)


def test_monte_carlo_reserved_lanes_match_main(reserved_run):
    res = run_mc(300, trucks_per_day=15000, seed=3, gate_lanes=RES_LANES,
                 al_khumra_flows=RES_FLOWS, al_khumra_lanes=RESERVED)["results"]
    qs = reserved_run["queue_stock"]
    for op in ("ENTRY", "EXIT"):
        q = qs[(qs.gate == "G9") & (qs.operation == op)]
        for col, main_value in ((f"peak_queue_G9_{op}", q.queue_stock.max()),
                                (f"peak_queue_G9_{op}_al_khumra", q.al_khumra_queue_stock.max())):
            lo, hi = res[col].quantile([0.01, 0.99])
            assert lo <= main_value <= hi, (col, main_value, lo, hi)


def test_random_empty_shortfall_is_handled_and_reported():
    # All DPW trucks via Al-Khumra with the terminal's own 30% empty share:
    # valid on average, but in some scenarios cargo draws leave the terminal a
    # few empties short. Those trucks are taken full and the shortfall counted.
    flows = {**ZERO_FLOWS, "DPW": {"mode": "share", "entry_value": 1.0, "exit_value": 1.0,
                                   "entry_gate": "G9", "exit_gate": "G9", "empty_share": 0.30}}
    lanes = {**model.GATE_LANES, "G9": {"entry": 3, "exit": 4}}
    r = run_main(trucks_per_day=10500, seed=3, gate_lanes=lanes, al_khumra_flows=flows)
    dpw = r["trucks"][r["trucks"].terminal == "DPW"]
    assert dpw.al_khumra_entry.all() and dpw.al_khumra_exit.all()
    needed = int(np.floor(0.30 * len(dpw) + 0.5))
    assert r["al_khumra_empty_shortfall"].get("DPW", 0) == needed - (dpw.cargo == "EMPTY").sum()
    res = run_mc(200, trucks_per_day=10500, seed=3, gate_lanes=lanes, al_khumra_flows=flows)["results"]
    # Selecting every DPW truck at the terminal's own empty share: the daily
    # draws almost never match exactly, so small shortfalls are reported.
    assert (res["al_khumra_empty_shortfall"] > 0).any()
    assert res["al_khumra_empty_shortfall"].max() < 0.05 * len(dpw)
    assert (res["al_khumra_trucks"] == len(dpw)).all()
