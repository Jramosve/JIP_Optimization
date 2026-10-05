import html
import random
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

import main as model
import monte_carlo

st.set_page_config(page_title="Jeddah Islamic Port - Traffic Model", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
/* ============================================================
   JIP V1 — Dark / Black dashboard theme
   ============================================================ */
:root {
    --bg: #0b0d10;
    --panel: #12161c;
    --panel-2: #171c23;
    --border: #2a3038;
    --text: #f2f4f7;
    --muted: #9aa3ad;
    --accent: #4f8cff;
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: rgba(11,13,16,0.96); }
.block-container { padding-top: 1.1rem; max-width: 1500px; }
h1, h2, h3, h4 { color: var(--text) !important; }
p, li, label, .stMarkdown, .stCaption { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* KPI cards */
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 11px 14px;
    box-shadow: none;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1014;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 { color: var(--text) !important; }

/* Inputs */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div,
.stTextInput > div > div,
.stSlider > div,
[data-baseweb="select"] > div {
    background: var(--panel-2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
[data-baseweb="popover"] { background: var(--panel-2) !important; }
[data-baseweb="menu"] { background: var(--panel-2) !important; }
[data-baseweb="menu"] li { color: var(--text) !important; }

/* Tables / dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 7px;
    overflow: hidden;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: #171c23;
    color: var(--text);
    border: 1px solid #343b45;
    border-radius: 6px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent);
    color: white;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Main title */
.jip-title { color: #ffffff; font-size: 2rem; font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0.05rem; }
.jip-subtitle { color: #8f98a3; font-size: 0.92rem; margin-bottom: 1rem; }
</style>
<div class="jip-title">Jeddah Islamic Port</div>
<div class="jip-subtitle">Truck Traffic & Gate Capacity Assessment · V1.1</div>""", unsafe_allow_html=True)
st.caption("Preliminary screening model. Gate, junction, road and demand assumptions should be calibrated with ELM/Mawani data before operational use.")

TERMINALS = ["RSGT", "DPW", "MPT", "LOGISTICS"]
CARGOS = ["FULL", "EMPTY"]
GATES = ["G1", "G4", "G6", "G8", "G9"]
JUNCTIONS = list(model.JUNCTIONS.keys())


def rules_to_df(rules):
    rows = []
    for terminal in TERMINALS:
        for cargo in CARGOS:
            row = {"Flow": f"{terminal} {cargo}", **{g: 0.0 for g in GATES}}
            for gate, share in rules[(terminal, cargo)]:
                row[gate] = share * 100.0
            rows.append(row)
    return pd.DataFrame(rows).set_index("Flow")


def df_to_rules(df):
    rules, errors = {}, []
    for flow, row in df.iterrows():
        terminal, cargo = flow.rsplit(" ", 1)
        values = {g: float(row.get(g, 0) or 0) for g in GATES}
        total = sum(values.values())
        if not np.isclose(total, 100.0, atol=0.05):
            errors.append(f"{flow}: shares sum to {total:.1f}%")
        rules[(terminal, cargo)] = [(g, values[g] / 100.0) for g in GATES if values[g] > 0]
    return rules, errors


def access_rules_to_df(rules):
    # Streamlit data_editor does not support MultiIndex dataframes.
    # Keep the terminal/cargo flow as a normal, read-only column instead.
    rows=[]
    for terminal in TERMINALS:
        for cargo in CARGOS:
            row={"Flow": f"{terminal} | {cargo}", **{g:0.0 for g in GATES}}
            for gate, share in rules[(terminal, cargo)]: row[gate]=share*100.0
            rows.append(row)
    return pd.DataFrame(rows)


def df_to_access_rules(df):
    rules, errors={},[]
    for _,row in df.iterrows():
        flow=str(row.get("Flow", "")).strip()
        if "|" in flow:
            terminal,cargo=[x.strip() for x in flow.split("|",1)]
        else:
            parts=flow.rsplit(" ",1)
            terminal,cargo=(parts[0],parts[1]) if len(parts)==2 else (flow,"")
        values={g:float(row.get(g,0) or 0) for g in GATES}
        total=sum(values.values())
        if not np.isclose(total,100.0,atol=0.05): errors.append(f"{terminal} {cargo}: access-gate shares sum to {total:.1f}%")
        rules[(terminal,cargo)]=[(g,values[g]/100.0) for g in GATES if values[g]>0]
    return rules, errors


def gate_service_df():
    rows=[
        {"Flow":"FULL — ENTRY","Processing time (sec)":model.GATE_TIMES_SEC["FULL_ENTRY"]},
        {"Flow":"FULL — EXIT","Processing time (sec)":model.GATE_TIMES_SEC["FULL_EXIT"]},
        {"Flow":"EMPTY — ENTRY","Processing time (sec)":model.GATE_TIMES_SEC["EMPTY_ENTRY"]},
        {"Flow":"EMPTY — EXIT","Processing time (sec)":model.GATE_TIMES_SEC["EMPTY_EXIT"]},
    ]
    return pd.DataFrame(rows).set_index("Flow")


def df_to_gate_service_times(df):
    times={"FULL_ENTRY":35.0,"FULL_EXIT":30.0,"EMPTY_ENTRY":20.0,"EMPTY_EXIT":20.0}
    mapping={"FULL — ENTRY":"FULL_ENTRY","FULL — EXIT":"FULL_EXIT","EMPTY — ENTRY":"EMPTY_ENTRY","EMPTY — EXIT":"EMPTY_EXIT"}
    for flow,row in df.iterrows():
        times[mapping[flow]]=float(row["Processing time (sec)"])
    return times


def terminal_share_df(shares):
    return pd.DataFrame([{"Terminal": k, "Share (%)": float(v)*100.0} for k,v in shares.items()])

def df_to_terminal_shares(df):
    values={str(r["Terminal"]): float(r["Share (%)"] or 0)/100.0 for _,r in df.iterrows()}
    total=sum(values.values())
    errors=[]
    if not np.isclose(total,1.0,atol=0.0005): errors.append(f"Terminal shares sum to {total*100:.1f}%")
    if any(v<0 for v in values.values()): errors.append("Terminal shares cannot be negative.")
    return values, errors

def cargo_share_df(shares):
    rows=[]
    for terminal in TERMINALS:
        rows.append({"Terminal":terminal,"Full (%)":float(shares[terminal]["FULL"])*100.0,"Empty (%)":float(shares[terminal]["EMPTY"])*100.0})
    return pd.DataFrame(rows)

def df_to_cargo_shares(df):
    values={}; errors=[]
    for _,r in df.iterrows():
        terminal=str(r["Terminal"]); full=float(r["Full (%)"] or 0)/100.0; empty=float(r["Empty (%)"] or 0)/100.0
        if not np.isclose(full+empty,1.0,atol=0.0005): errors.append(f"{terminal}: Full + Empty must equal 100%.")
        values[terminal]={"FULL":full,"EMPTY":empty}
    return values, errors

def terminal_process_df(process):
    rows=[]
    for terminal in TERMINALS:
        rows.append({"Terminal":terminal,"Full (min)":float(process[terminal]["FULL"]),"Empty (min)":float(process[terminal]["EMPTY"])})
    return pd.DataFrame(rows)

def df_to_terminal_process(df):
    return {str(r["Terminal"]): {"FULL":float(r["Full (min)"]),"EMPTY":float(r["Empty (min)"])} for _,r in df.iterrows()}


def scenario_rules(name):
    access = {k:list(v) for k,v in model.ENTRY_ACCESS_RULES.items()}
    exit_ = {k:list(v) for k,v in model.EXIT_GATE_RULES.items()}
    if name == "RSGT All → G1":
        access[("RSGT","FULL")]=[("G1",1.0)]
        access[("RSGT","EMPTY")]=[("G1",1.0)]
    return access, exit_


def los_from_vc(vc):
    vc=float(vc)
    if vc <= 0.15: return "A"
    if vc <= 0.27: return "B"
    if vc <= 0.43: return "C"
    if vc <= 0.64: return "D"
    if vc <= 1.00: return "E"
    return "F"


def vc_to_hex(vc):
    return {"A":"#5caf3a","B":"#9ac34a","C":"#f0a23a","D":"#e76f2f","E":"#d73027","F":"#7f1d1d"}[los_from_vc(vc)]


def queue_to_hex(q):
    if q < 25: return "#22a06b"
    if q < 75: return "#e0b100"
    if q < 150: return "#f28c28"
    return "#d64545"


def build_route_editor_graph():
    routes = model.read_kml_routes()
    return model.build_network(routes)


def configured_od_rows(graph, entry_rules, exit_rules):
    rows=[]
    seen=set()
    # ENTRY access allocation is terminal + cargo specific.
    for (terminal, cargo), options in entry_rules.items():
            destinations = (
                list(model.MPT_DESTINATION_SPLIT.keys()) if terminal == "MPT"
                else list(model.LOGISTICS_DESTINATION_DAILY.keys()) if terminal == "LOGISTICS"
                else [model.get_destination_node(terminal)]
            )
            for gate, share in options:
                if float(share) <= 0: continue
                for destination in destinations:
                    end = model.get_destination_node(terminal, destination) if terminal == "MPT" else destination
                    key=("ENTRY",terminal,cargo,gate,end)
                    if key in seen: continue
                    seen.add(key)
                    try:
                        path=nx.shortest_path(graph,gate,end,weight="travel_time_min"); valid=True; error=""
                    except nx.NetworkXNoPath:
                        path=[gate,end]; valid=False; error="No route"
                    rows.append({"Direction":"ENTRY","Flow":f"{terminal} {cargo}","OD":f"{gate} → {end}","Path":" > ".join(path),"Valid":valid,"Error":error,"_key":key})

    # EXIT allocation remains cargo-specific.
    for (terminal,cargo), options in exit_rules.items():
        destinations=(list(model.MPT_DESTINATION_SPLIT.keys()) if terminal=="MPT" else list(model.LOGISTICS_DESTINATION_DAILY.keys()) if terminal=="LOGISTICS" else [model.get_destination_node(terminal)])
        for gate,share in options:
            if float(share)<=0: continue
            for destination in destinations:
                start=model.get_destination_node(terminal,destination) if terminal=="MPT" else destination
                key=("EXIT",terminal,cargo,start,gate)
                if key in seen: continue
                seen.add(key)
                try:
                    path=nx.shortest_path(graph,start,gate,weight="travel_time_min"); valid=True; error=""
                except nx.NetworkXNoPath:
                    path=[start,gate]; valid=False; error="No route"
                rows.append({"Direction":"EXIT","Flow":f"{terminal} {cargo}","OD":f"{start} → {gate}","Path":" > ".join(path),"Valid":valid,"Error":error,"_key":key})
    return pd.DataFrame(rows)


def parse_route_overrides(route_df, graph):
    overrides = {}
    errors = []
    for _, row in route_df.iterrows():
        direction, terminal, cargo, start, end = row["_key"]
        raw = str(row.get("Path", "")).replace("→", ">").replace("->", ">")
        path = [x.strip() for x in raw.split(">") if x.strip()]
        if not path or path[0] != start or path[-1] != end:
            errors.append(f"{direction} {terminal} {cargo}: path must start at {start} and finish at {end}.")
            continue
        missing = [f"{a} → {b}" for a, b in zip(path[:-1], path[1:]) if not graph.has_edge(a, b)]
        if missing:
            errors.append(f"{direction} {terminal} {cargo} ({start} → {end}): missing link(s): {', '.join(missing)}")
            continue
        overrides[(start, end)] = path
    return overrides, errors


def gate_wait_summary(gates):
    rows=[]
    for gate in GATES:
        for operation in ["ENTRY", "EXIT"]:
            g=gates[(gates.gate==gate)&(gates.operation==operation)] if not gates.empty else pd.DataFrame()
            rows.append({
                "Gate":gate, "Operation":operation,
                "Trucks":int(len(g)),
                "Avg queue time (min)":float(g.wait_min.mean()) if len(g) else 0.0,
                "Peak queue time (min)":float(g.wait_min.max()) if len(g) else 0.0,
                "Processing time (sec)":float(g.service_sec.mean()) if len(g) else 0.0,
            })
    return pd.DataFrame(rows)


def result_kpis(result):
    trucks=result["trucks"]; gates=result["gates"]; roads=result["roads"]
    hourly=(trucks.assign(hour=(trucks.arrival_min//60).astype(int)).groupby("hour").size() if not trucks.empty else pd.Series(dtype=float))
    return {
        "Daily truck movements": int(len(trucks)),
        "Peak hourly demand": int(hourly.max()) if len(hourly) else 0,
        "Avg turnaround (min)": float(trucks.total_time_min.mean()) if len(trucks) else 0.0,
        "Avg gate queue (min)": float(gates.wait_min.mean()) if len(gates) else 0.0,
        "Peak gate queue (min)": float(gates.wait_min.max()) if len(gates) else 0.0,
        "Peak road V/C": float(roads.peak_vc.max()) if len(roads) else 0.0,
    }


def build_network_map(graph, road_hourly, queue_stock, hour, gates_df=None):
    # Include every graph node, including explicit KML Point nodes that are
    # not yet connected to a road. This keeps the map aligned with Google Earth.
    nodes = [
        (n, float(d["lon"]), float(d["lat"]))
        for n, d in graph.nodes(data=True)
        if "lat" in d and "lon" in d
    ]
    if not nodes:
        return "<p>No network coordinates available.</p>"

    geometry_points = []
    for _, _, data in graph.edges(data=True):
        geometry_points.extend(data.get("geometry", []))
    all_points = [(lon, lat) for _, lon, lat in nodes] + geometry_points
    lons = [p[0] for p in all_points]
    lats = [p[1] for p in all_points]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lon_pad = max((max_lon-min_lon)*0.20, 0.0015)
    lat_pad = max((max_lat-min_lat)*0.20, 0.0015)
    min_lon -= lon_pad; max_lon += lon_pad
    min_lat -= lat_pad; max_lat += lat_pad

    W, H = 1250, 720
    def xy(lon, lat):
        return ((lon-min_lon)/(max_lon-min_lon)*W,
                H-(lat-min_lat)/(max_lat-min_lat)*H)

    selected = {}
    if not road_hourly.empty:
        h = road_hourly[road_hourly.hour == hour]
        for physical, g in h.groupby("physical_road"):
            selected[physical] = float(g.vc.max())

    gate_q = {}
    gate_wait = {}
    if not queue_stock.empty:
        hq = queue_stock[queue_stock.hour == hour]
        for gate in GATES:
            vals = hq[hq.gate == gate]["queue_stock"]
            gate_q[gate] = float(vals.max()) if len(vals) else 0.0
    if gates_df is not None and not gates_df.empty:
        gh = gates_df[(gates_df["requested_min"] // 60).astype(int) == int(hour)]
        for gate in GATES:
            g=gh[gh.gate==gate]
            gate_wait[gate]=(float(g.wait_min.mean()) if len(g) else 0.0, float(g.wait_min.max()) if len(g) else 0.0)
    else:
        gate_wait={g:(0.0,0.0) for g in GATES}

    # Exit-gate spillback is shown directly on the final approach link.
    spillback_links = {}
    for gate in ["G1", "G8", "G9"]:
        candidates = [(a, b, d) for a, b, d in graph.edges(data=True) if b == gate]
        if not candidates:
            continue
        a, b, d = candidates[0]
        q = gate_q.get(gate, 0.0)
        storage = max(1.0, float(d.get("capacity_vph", 0.0)) * float(d.get("travel_time_min", 0.0)) / 60.0)
        if q > storage:
            spillback_links[d["physical_road"]] = (q, storage, gate)

    # Some KML segments form one continuous visual corridor. Keep them as
    # separate network links for capacity/flow calculations, but use a common
    # colour on the map so the corridor is not perceived as two overlapping
    # roads. The G8 approach is N6 -> N9 -> G8.
    visual_corridors = {
        "G8_APPROACH": {"N6 - G8", "N9 - G8"},
    }

    physical_to_visual_group = {}
    visual_group_vc = {}
    visual_group_spill = {}
    for a, b, data in graph.edges(data=True):
        physical = data["physical_road"]
        kml_name = data.get("kml_name", "")
        for group_name, members in visual_corridors.items():
            if kml_name in members:
                physical_to_visual_group[physical] = group_name
                break

    for group_name in visual_corridors:
        members_physical = [
            physical for physical, group in physical_to_visual_group.items()
            if group == group_name
        ]
        if members_physical:
            visual_group_vc[group_name] = max(
                selected.get(physical, 0.0) for physical in members_physical
            )
            visual_group_spill[group_name] = any(
                physical in spillback_links for physical in members_physical
            )

    road_svg = []
    seen = set()
    for a, b, data in graph.edges(data=True):
        physical = data["physical_road"]
        if physical in seen:
            continue
        seen.add(physical)
        pts = [f"{xy(lon,lat)[0]:.1f},{xy(lon,lat)[1]:.1f}" for lon,lat in data["geometry"]]
        vc = selected.get(physical, 0.0)
        visual_group = physical_to_visual_group.get(physical)
        display_vc = visual_group_vc.get(visual_group, vc)
        los = los_from_vc(display_vc)
        color = vc_to_hex(display_vc)
        width = 9 if display_vc > 1 else 7
        spill_txt = ""
        if physical in spillback_links:
            q, storage, gate = spillback_links[physical]
            color = "#7f1d1d"; width = 11
            spill_txt = f" | SPILLBACK at {gate}: {q:.0f} trucks > {storage:.0f} storage"
        elif visual_group and visual_group_spill.get(visual_group, False):
            spill_txt = " | G8 approach corridor spillback"
        corridor_txt = (
            f" | Corridor V/C {display_vc:.2f}"
            if visual_group else ""
        )
        road_svg.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">'
            f'<title>{html.escape(physical)} | Segment V/C {vc:.2f} | LOS {los}{corridor_txt}{spill_txt}</title></polyline>'
        )

    node_svg = []
    gate_set = set(GATES)
    terminal_set = {"RSGT", "DPW", "MPT1", "MPT2", "MPT3", "Logipoint"}
    for node, lon, lat in nodes:
        x, y = xy(lon, lat)
        if node in gate_set:
            q = gate_q.get(node, 0)
            color = queue_to_hex(q)
            radius = 10 + min(q/25, 8)
            avg_w, peak_w = gate_wait.get(node, (0.0, 0.0))
            title = f"{node} | queue {q:.0f} trucks | avg wait {avg_w:.1f} min | peak wait {peak_w:.1f} min"
        elif node in JUNCTIONS:
            color = "#7b61a8"; radius = 8; title = f"{node} | junction"
        elif node in terminal_set:
            color = "#4b87c5"; radius = 7; title = node
        else:
            color = "#555b66"; radius = 6; title = node
        node_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" '
            f'stroke="white" stroke-width="2"><title>{html.escape(title)}</title></circle>'
            f'<text x="{x+12:.1f}" y="{y-10:.1f}" class="label">{html.escape(node)}</text>'
        )

    return f'''<style>
    .mapwrap{{background:#10131a;border:1px solid #2a2f3a;border-radius:10px;padding:12px}}
    .maptitle{{color:#f2f4f7;font:600 15px Arial;margin-bottom:6px}}
    .legend{{display:flex;flex-wrap:wrap;gap:12px;color:#b9c0cc;font:12px Arial;margin:8px 0 2px}}
    .label{{fill:#111827;font:600 12px Arial;paint-order:stroke;stroke:#FFFFFF;stroke-width:3px}}
    svg{{width:100%;height:auto;display:block}}
    </style>
    <div class="mapwrap"><div class="maptitle">Network pressure — {hour:02d}:00 to {hour:02d}:59</div>
    <svg viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" rx="8" fill="#F3F6F8"/>{''.join(road_svg)}{''.join(node_svg)}</svg>
    <div class="legend">
    <span style="color:#5caf3a">A ≤0.15</span><span style="color:#9ac34a">B 0.15–0.27</span>
    <span style="color:#f0a23a">C 0.27–0.43</span><span style="color:#e76f2f">D 0.43–0.64</span>
    <span style="color:#d73027">E 0.64–1.00</span><span style="color:#7f1d1d">F &gt;1.00 / congestion</span>
    <span>Gate circles = queue stock</span><span>Gate tooltip = queue time</span></div></div>'''

with st.sidebar:
    st.header("Scenario")
    mode=st.radio("Mode",["Basic","Advanced"],horizontal=True)
    presets={"Base — 10,500":10500,"Mid-term — 15,000":15000,"Long-term — 20,000":20000,"Hypothetical — 30,000":30000,"Custom":None}
    preset=st.selectbox("Demand scenario",list(presets))
    default_trucks=presets[preset] or 10500
    trucks_per_day=st.slider("Truck movements / day",5000,30000,int(default_trucks),500) if preset=="Custom" else default_trucks
    hour=st.slider("Hour of day",0,23,21)
    route_randomization_pct=st.slider("Route randomization",0,100,0,5)
    arrival_time_randomization_pct=st.slider("Hourly profile variability",0,25,0,5,help="Base case uses the observed hourly profile exactly. Monte Carlo can introduce controlled variability.")
    seed=st.number_input("Random seed",1,999999,42,1)
    compare_base=st.checkbox("Compare with Base — 10,500", value=True)
    if st.button("Reset scenario"):
        st.session_state.clear(); st.rerun()
    st.divider()
    st.header("Appointment management")
    management_enabled=st.checkbox("Enable ELM appointment cap",value=True,help="Port-entry release is capped at the ELM hourly limit. Excess demand rolls forward FIFO; no trucks are dropped.")
    max_entries=st.number_input("ELM maximum entries / hour",100,3000,500,50,disabled=not management_enabled)
    st.caption("Base assumption: 500 port entries/hour. Released trucks are distributed approximately uniformly within each hour. Internal movements = 5% of gate truck traffic and do not use Mawani gates.")
    if mode=="Advanced":
        st.divider(); st.header("Network inputs")
        average_speed=st.number_input("Average road speed (km/h)",5.0,80.0,30.0,1.0)
        lanes_per_direction=st.number_input("Default road lanes / direction",1,8,2,1)
        capacity_per_lane=st.number_input("Road capacity / lane / hour",300,1800,800,50)
        n2_n1_lanes=st.number_input("N4–N2 corridor lanes / direction",3,4,3,1)
        junctions={k:dict(v) for k,v in model.JUNCTIONS.items()}
        for node in sorted(junctions.keys()):
            junctions[node]["capacity_vph"]=st.number_input(f"{node} capacity (vph)",100,3000,700,50,key=f"{node}_cap")
            junctions[node]["maneuver_delay_sec"]=st.number_input(f"{node} maneuver delay (sec)",0.0,60.0,float(junctions[node].get("maneuver_delay_sec",0.0)),1.0,key=f"{node}_delay")
        gate_lanes={g:dict(v) for g,v in model.GATE_LANES.items()}
        st.subheader("Gate lanes")
        for gate in GATES:
            c1,c2=st.columns(2)
            gate_lanes[gate]["entry"]=c1.number_input(f"{gate} IN",0,20,int(gate_lanes[gate]["entry"]),1,key=f"{gate}_in")
            gate_lanes[gate]["exit"]=c2.number_input(f"{gate} OUT",0,20,int(gate_lanes[gate]["exit"]),1,key=f"{gate}_out")
    else:
        average_speed=30.0; lanes_per_direction=2; capacity_per_lane=800; n2_n1_lanes=3
        junctions={k:dict(v) for k,v in model.JUNCTIONS.items()}
        gate_lanes={g:dict(v) for g,v in model.GATE_LANES.items()}

# Base flow inputs are editable in the main panel.
st.header("Flow & operational inputs")
st.caption("Base values reflect the latest JIP operational information. Percentages are editable for future expansion and reconfiguration scenarios.")
flow_errors=[]
fc1,fc2=st.columns(2)
with fc1:
    st.subheader("Terminal split")
    ts=st.data_editor(terminal_share_df(model.TERMINAL_SHARES),use_container_width=True,num_rows="fixed",hide_index=True,column_config={"Share (%)":st.column_config.NumberColumn(min_value=0,max_value=100,step=1)},key="terminal_share_editor")
    terminal_shares,err=df_to_terminal_shares(ts); flow_errors+=err
with fc2:
    st.subheader("Full / Empty split")
    cs=st.data_editor(cargo_share_df(model.CARGO_SHARES),use_container_width=True,num_rows="fixed",hide_index=True,column_config={"Full (%)":st.column_config.NumberColumn(min_value=0,max_value=100,step=1),"Empty (%)":st.column_config.NumberColumn(min_value=0,max_value=100,step=1)},key="cargo_share_editor")
    cargo_shares,err=df_to_cargo_shares(cs); flow_errors+=err

st.subheader("Gate allocation")
col1,col2=st.columns(2)
with col1:
    st.caption("Entry allocation — terminal × cargo. Each row must sum to 100%.")
    ed=st.data_editor(access_rules_to_df(model.ENTRY_ACCESS_RULES),use_container_width=True,num_rows="fixed",column_config={g:st.column_config.NumberColumn(g,min_value=0,max_value=100,step=5) for g in GATES},key="entry_access_editor")
    entry_rules,entry_errors=df_to_access_rules(ed); flow_errors+=entry_errors
with col2:
    st.caption("Exit allocation — terminal × cargo. Each row must sum to 100%.")
    xd=st.data_editor(rules_to_df(model.EXIT_GATE_RULES),use_container_width=True,num_rows="fixed",column_config={g:st.column_config.NumberColumn(g,min_value=0,max_value=100,step=5) for g in GATES},key="exit_editor")
    exit_rules,exit_errors=df_to_rules(xd); flow_errors+=exit_errors

if mode=="Advanced":
    st.subheader("Processing times")
    tc1,tc2=st.columns(2)
    with tc1:
        st.caption("Gate processing — independent of terminal.")
        gt=st.data_editor(gate_service_df(),use_container_width=True,num_rows="fixed",column_config={"Processing time (sec)":st.column_config.NumberColumn(min_value=1,max_value=300,step=1)},key="gate_service_times_editor")
        gate_times=df_to_gate_service_times(gt)
    with tc2:
        st.caption("Terminal processing — same Full/Empty time by terminal unless edited.")
        tp=st.data_editor(terminal_process_df(model.TERMINAL_PROCESS_MIN),use_container_width=True,num_rows="fixed",hide_index=True,column_config={"Full (min)":st.column_config.NumberColumn(min_value=1,max_value=300,step=1),"Empty (min)":st.column_config.NumberColumn(min_value=1,max_value=300,step=1)},key="terminal_process_editor")
        terminal_process= df_to_terminal_process(tp)
else:
    gate_times=dict(model.GATE_TIMES_SEC); terminal_process={k:dict(v) for k,v in model.TERMINAL_PROCESS_MIN.items()}

# Keep terminal process editable only in Advanced; apply it temporarily to the model run.
model_terminal_process_original={k:dict(v) for k,v in model.TERMINAL_PROCESS_MIN.items()}
model.TERMINAL_PROCESS_MIN=terminal_process

# Automatic route configuration remains available as an Advanced diagnostic.
route_overrides={}; route_errors=[]
if mode=="Advanced":
    with st.expander("Internal route configuration",expanded=False):
        try:
            editor_graph=build_route_editor_graph(); route_defaults=configured_od_rows(editor_graph,entry_rules,exit_rules)
            route_editor=st.data_editor(route_defaults[["Direction","Flow","OD","Path","Valid"]],use_container_width=True,hide_index=True,disabled=["Direction","Flow","OD","Valid"],column_config={"Path":st.column_config.TextColumn("Path (editable)",width="large"),"Valid":st.column_config.CheckboxColumn("Valid",disabled=True)},key="route_editor")
            route_overrides,route_errors=parse_route_overrides(route_editor.assign(_key=route_defaults.loc[route_editor.index,"_key"].values),editor_graph)
            flow_errors+=route_errors
        except Exception as exc:
            flow_errors.append(f"Could not build route editor: {exc}")

if flow_errors:
    st.error("Please correct the following inputs before running:\n\n" + "\n".join(flow_errors))
run_ok=not flow_errors

# Inputs shared by the simulation run and the gate Monte Carlo. Used to flag
# when the Monte Carlo is compared against a run made with other inputs.
scenario_signature=repr((int(trucks_per_day),arrival_time_randomization_pct,bool(management_enabled),int(max_entries),
    sorted(entry_rules.items()),sorted(exit_rules.items()),sorted((g,sorted(v.items())) for g,v in gate_lanes.items()),
    sorted(gate_times.items()),sorted(terminal_shares.items()),sorted((k,sorted(v.items())) for k,v in cargo_shares.items()),
    float(average_speed),sorted((k,sorted(v.items())) for k,v in junctions.items()),
    sorted((k,sorted(v.items())) for k,v in terminal_process.items()),sorted(route_overrides.items())))

if "result" not in st.session_state: st.session_state.result=None
if "result_signature" not in st.session_state: st.session_state.result_signature=None
if "base_result" not in st.session_state: st.session_state.base_result=None
if "mc_result" not in st.session_state: st.session_state.mc_result=None

if st.button("Run simulation",type="primary",disabled=not run_ok):
    with st.spinner("Running JIP traffic simulation..."):
        st.session_state.result=model.run_simulation(
            trucks_per_day=trucks_per_day,route_randomization=route_randomization_pct/100,
            arrival_time_variability=arrival_time_randomization_pct/100,seed=int(seed),
            entry_gate_rules=entry_rules,exit_gate_rules=exit_rules,entry_access_rules=entry_rules,gate_lanes=gate_lanes,
            gate_times_sec=gate_times,n2_n1_corridor_lanes=n2_n1_lanes,
            average_speed_kmh=average_speed,lanes_per_direction=lanes_per_direction,
            capacity_per_lane_vph=capacity_per_lane,junctions=junctions,
            appointment_management_enabled=management_enabled,max_entries_per_hour=int(max_entries),
            route_overrides=route_overrides,logistics_destination_daily=None,terminal_shares=terminal_shares,cargo_shares=cargo_shares,save_outputs=False)
        if compare_base:
            st.session_state.base_result=model.run_simulation(
                trucks_per_day=10500,route_randomization=0.0,arrival_time_variability=0.0,seed=int(seed),
                entry_gate_rules=model.ENTRY_ACCESS_RULES,exit_gate_rules=model.EXIT_GATE_RULES,entry_access_rules=model.ENTRY_ACCESS_RULES,
                gate_lanes={g:dict(v) for g,v in model.GATE_LANES.items()},gate_times_sec=dict(model.GATE_TIMES_SEC),
                n2_n1_corridor_lanes=model.N2_N1_CORRIDOR_LANES,average_speed_kmh=model.AVERAGE_SPEED_KMH,
                lanes_per_direction=model.LANES_PER_DIRECTION,capacity_per_lane_vph=model.CAPACITY_PER_LANE_VPH,
                junctions={k:dict(v) for k,v in model.JUNCTIONS.items()},appointment_management_enabled=management_enabled,
                max_entries_per_hour=int(max_entries),route_overrides={},
                logistics_destination_daily=None,terminal_shares={k:0.40 if k=="RSGT" else 0.35 if k=="DPW" else 0.20 if k=="MPT" else 0.05 for k in TERMINALS},cargo_shares={k:{"FULL":0.70,"EMPTY":0.30} for k in TERMINALS},save_outputs=False)
        else:
            st.session_state.base_result=None
    st.session_state.result_signature=scenario_signature
    st.session_state.mc_result=None

result=st.session_state.result
if result is None:
    st.info("Set the scenario parameters and click Run simulation.")
    st.stop()

trucks=result["trucks"].copy(); gates=result["gates"].copy(); roads=result["roads"].copy(); road_hourly=result["road_hourly"].copy(); queues=result["queues"].copy(); queue_stock=result.get("queue_stock",queues).copy(); junction_results=result["junctions"].copy(); junction_summary=result.get("junction_summary",pd.DataFrame()).copy(); spillback=result.get("spillback",pd.DataFrame()).copy(); graph=result["graph"]; appointment=result.get("appointment_profile",pd.DataFrame())

hour_q=queue_stock[queue_stock.hour==hour]
queue_in={g:int(hour_q[(hour_q.gate==g)&(hour_q.operation=="ENTRY")].queue_stock.max()) if len(hour_q[(hour_q.gate==g)&(hour_q.operation=="ENTRY")]) else 0 for g in GATES}
queue_out={g:int(hour_q[(hour_q.gate==g)&(hour_q.operation=="EXIT")].queue_stock.max()) if len(hour_q[(hour_q.gate==g)&(hour_q.operation=="EXIT")]) else 0 for g in GATES}
avg_time=float(trucks.total_time_min.mean()); max_queue=max(list(queue_in.values())+list(queue_out.values()) or [0]); peak_vc=float(roads.peak_vc.max()) if not roads.empty else 0.0
avg_gate_wait=float(gates.wait_min.mean()) if not gates.empty else 0.0
peak_gate_wait=float(gates.wait_min.max()) if not gates.empty else 0.0

st.subheader(f"Network status — {hour:02d}:00")
c=st.columns(7)
internal_moves=int(round(len(trucks)*model.INTERNAL_MOVEMENTS_SHARE))
kpi_values=[("Daily gate truck movements",f"{len(trucks):,}"),("Internal movements (5%)",f"{internal_moves:,}"),("Peak hourly demand",f"{int(trucks.assign(hour=(trucks.arrival_min//60).astype(int)).groupby('hour').size().max()):,}"),("Avg turnaround",f"{avg_time:.1f} min"),("Avg gate queue",f"{avg_gate_wait:.1f} min"),("Peak gate queue",f"{peak_gate_wait:.1f} min"),("Peak road V/C",f"{peak_vc:.2f}")]
for col,(label,value) in zip(c,kpi_values): col.metric(label,value)

if management_enabled and not appointment.empty:
    st.subheader("Appointment management")
    st.dataframe(appointment,use_container_width=True)
    st.caption("Uncovered demand is the demand that could not be released through the port-wide appointment cap and is carried into the next hour.")

st.subheader("Exit-gate spillback")
if not spillback.empty:
    st.dataframe(spillback,use_container_width=True,hide_index=True)
    st.caption("Approach storage represents the approximate physical truck storage on the final exit-gate approach link. When full, upstream junctions can be held by queued exit traffic, reproducing spillback into the internal road network.")

st.subheader("Network congestion map")
components.html(build_network_map(graph,road_hourly,queue_stock,hour,gates),height=690,scrolling=False)
st.caption("Road colours use the V/C LOS bands from the provided reference table as a screening classification: A/B/C/D/E/F = ≤0.15 / 0.27 / 0.43 / 0.64 / 1.00 / >1.00. The model capacity itself remains the explicit lane-capacity assumption; the reference table is not used to extrapolate capacity from 60 km/h to 30 km/h. Gate circles represent queue stock. IN and OUT gate lanes are separate resources.")

if st.session_state.base_result is not None:
    st.subheader("Base vs Scenario")
    base_k=result_kpis(st.session_state.base_result); scen_k=result_kpis(result)
    rows=[]
    for key in base_k:
        b=base_k[key]; v=scen_k[key]; delta=v-b
        rows.append({"KPI":key,"Base":b,"Scenario":v,"Δ":delta})
    comp=pd.DataFrame(rows)
    st.dataframe(comp,use_container_width=True,hide_index=True,column_config={"Base":st.column_config.NumberColumn(format="%.2f"),"Scenario":st.column_config.NumberColumn(format="%.2f"),"Δ":st.column_config.NumberColumn(format="%+.2f")})


st.subheader("Configured truck paths")
path_cols=[c for c in ["truck_id","terminal","cargo","entry_gate","network_destination","entry_path","exit_gate","exit_path"] if c in trucks.columns]
st.dataframe(trucks[path_cols].head(200),use_container_width=True)
st.caption("This table validates the actual route used by each simulated truck. Internal road capacity is assessed separately through the directional V/C results below.")

st.subheader("Gate queue profile")
qp=queue_stock.pivot_table(index="time_min",columns=["gate","operation"],values="queue_stock",aggfunc="max").fillna(0)
expected_cols=pd.MultiIndex.from_product([GATES,["ENTRY","EXIT"]],names=["gate","operation"])
qp=qp.reindex(columns=expected_cols,fill_value=0)
qp.index=qp.index/60.0
qp.columns=[f"{gate} {operation}" for gate,operation in qp.columns]
st.line_chart(qp)
st.subheader("Gate queue time by access point")
wait_df=gate_wait_summary(gates)
st.dataframe(wait_df,use_container_width=True,hide_index=True)
st.caption("Gate queue stock includes trucks waiting at the gate and trucks delayed on the upstream road while serving that gate; the same truck is counted only once. Queue time is the simulated waiting time before gate processing and is separate from gate processing time and terminal turnaround.")


st.subheader("Monte Carlo — Gate congestion")
st.caption("Fast gate model: reproduces main.py demand, hourly profile, ELM appointment cap, cargo mix, gate allocation, gate lanes and processing times, and the truck cycle (entry gate → route → terminal → route → exit gate). Road-link and junction queueing are not simulated, so results can diverge from the full simulation only when the internal network itself is saturated.")
mc1,mc2,mc3=st.columns(3)
mc_sims=mc1.number_input("Number of Monte Carlo scenarios",100,20000,1000,100)
mc_share_var_pct=mc2.slider("Terminal / cargo share variability (%)",0,20,0,1,help="0% = same terminal and Full/Empty shares as the simulation run; variability then comes only from the hourly profile and random cargo/gate choices, exactly as in main.py.")
mc3.metric("Hourly profile variability",f"{arrival_time_randomization_pct}%",help="Taken from the sidebar so the Monte Carlo describes the same scenario as the simulation run.")
if st.button("Run Gate Monte Carlo",disabled=not run_ok):
    progress=st.progress(0.0,text="Running gate-congestion scenarios...")
    st.session_state.mc_result=monte_carlo.run_monte_carlo(
        model,
        int(mc_sims),
        int(trucks_per_day),
        arrival_time_randomization_pct/100,
        mc_share_var_pct/100,
        management_enabled,
        int(max_entries),
        int(seed),
        entry_access_rules=entry_rules,
        exit_gate_rules=exit_rules,
        gate_lanes=gate_lanes,
        gate_times_sec=gate_times,
        terminal_shares=terminal_shares,
        cargo_shares=cargo_shares,
        average_speed_kmh=average_speed,
        junctions=junctions,
        terminal_process_min=terminal_process,
        route_overrides=route_overrides,
        graph=graph,
        progress_callback=lambda done,total: progress.progress(done/total,text=f"Running gate-congestion scenarios... {done:,}/{total:,}"),
    )
    st.session_state.mc_signature=scenario_signature
    progress.empty()

mc=st.session_state.mc_result
if mc:
    m=mc.get("metrics", {})
    results=mc.get("results", pd.DataFrame()).copy()
    cfg_mc=mc.get("configuration", {})
    percentile_labels=list(monte_carlo.PERCENTILES)
    comparable=st.session_state.get("mc_signature")==st.session_state.result_signature
    if not comparable:
        st.warning("The simulation run above used different inputs from this Monte Carlo. Re-run the simulation to compare them.")

    # Values from the current simulation run, using the same definitions as the
    # Monte Carlo: queue stock = trucks waiting + in service, sampled every
    # 5 minutes until 24:00; queue time = wait before gate processing.
    run_values={}
    qs_day=queue_stock[queue_stock.time_min<=24*60]
    for g in GATES:
        for op in ["ENTRY","EXIT"]:
            q=qs_day[(qs_day.gate==g)&(qs_day.operation==op)].queue_stock
            w=gates[(gates.gate==g)&(gates.operation==op)].wait_min
            run_values[f"peak_queue_{g}_{op}"]=float(q.max()) if len(q) else 0.0
            run_values[f"avg_wait_{g}_{op}"]=float(w.mean()) if len(w) else 0.0
            run_values[f"peak_wait_{g}_{op}"]=float(w.max()) if len(w) else 0.0
    run_values["peak_queue_all"]=max(v for k,v in run_values.items() if k.startswith("peak_queue_"))
    run_values["avg_wait_all"]=float(gates.wait_min.mean()) if not gates.empty else 0.0
    run_values["peak_wait_all"]=float(gates.wait_min.max()) if not gates.empty else 0.0

    st.caption(f'{cfg_mc.get("simulations",0):,} scenarios · {cfg_mc.get("trucks_per_day",0):,} trucks/day · hourly variability {cfg_mc.get("arrival_variability",0)*100:.0f}% · share variability {cfg_mc.get("demand_variability",0)*100:.0f}% · ELM cap {"on, "+format(cfg_mc.get("max_entries_per_hour",0),",")+"/h" if cfg_mc.get("management_enabled") else "off"}')

    for title,key,unit,fmt in [("Peak gate queue — all gates","peak_queue_all","trucks","{:.0f}"),("Peak gate queue time","peak_wait_all","min","{:.1f}"),("Average gate queue time","avg_wait_all","min","{:.1f}")]:
        st.markdown(f"**{title}**")
        cols=st.columns(6)
        vals=m.get(key,{})
        for i,label in enumerate(percentile_labels):
            cols[i].metric(label,f"{fmt.format(float(vals.get(label,0.0)))} {unit}")
        cols[5].metric("Simulation run",f"{fmt.format(run_values[key])} {unit}")

    st.subheader("Gate percentiles vs simulation run")
    summary_rows=[]
    dist_options={"All gates — Peak queue (trucks)":"peak_queue_all","All gates — Peak queue time (min)":"peak_wait_all","All gates — Avg queue time (min)":"avg_wait_all"}
    for g in GATES:
        for op in ["ENTRY","EXIT"]:
            lanes=int(gate_lanes.get(g,{}).get(op.lower(),0))
            if lanes<=0: continue
            for metric_key,metric_name in [("peak_queue","Peak queue (trucks)"),("peak_wait","Peak queue time (min)"),("avg_wait","Avg queue time (min)")]:
                col=f"{metric_key}_{g}_{op}"
                vals=m.get(col,{})
                dist_options[f"{g} {op} — {metric_name}"]=col
                summary_rows.append({"Gate":f"{g} {op}","Lanes":lanes,"Metric":metric_name,
                                     **{p:float(vals.get(p,0.0)) for p in percentile_labels},
                                     "Simulation run":run_values.get(col,0.0)})
    st.dataframe(pd.DataFrame(summary_rows),use_container_width=True,hide_index=True,
                 column_config={c:st.column_config.NumberColumn(format="%.1f") for c in percentile_labels+["Simulation run"]})
    st.caption("Queue = trucks waiting plus in service at the gate (same definition as the gate queue profile above). The simulation run is a single realisation and is expected to fall inside the P10–P90 band most of the time.")

    # ------------------------------------------------------------
    # Distribution selector
    # ------------------------------------------------------------
    st.subheader("Monte Carlo output distribution")
    selected_label=st.selectbox("Output",list(dist_options),index=0,key="mc_distribution_output")
    value_col=dist_options[selected_label]
    values=pd.to_numeric(results[value_col],errors="coerce").dropna() if value_col in results.columns else pd.Series(dtype=float)

    if not values.empty:
        percentiles={p:float(values.quantile(q)) for p,q in monte_carlo.PERCENTILES.items()}

        vmin,vmax=float(values.min()),float(values.max())
        edges=np.array([vmin-0.5,vmax+0.5]) if np.isclose(vmin,vmax) else np.linspace(vmin,vmax,31)
        counts,edges=np.histogram(values.to_numpy(),bins=edges)
        centers=(edges[:-1]+edges[1:])/2.0

        fig=go.Figure()
        fig.add_trace(go.Bar(
            x=centers,y=counts,width=np.diff(edges)*0.92,
            marker_color="#4f8cff",marker_line_width=0,
            hovertemplate=f"{html.escape(selected_label)}: %{{x:.1f}}<br>Frequency: %{{y}}<extra></extra>",
            name="Frequency",
        ))
        percentile_colors={"P10":"#f59e0b","P50":"#7c8ea3","P90":"#94a3b8","P95":"#6ee7b7","P99":"#67e8f9"}
        for p,value in percentiles.items():
            fig.add_vline(x=value,line_width=2,line_dash="dash",line_color=percentile_colors[p],
                          annotation_text=f"{p}: {value:.1f}",annotation_position="top",annotation_font_color=percentile_colors[p])
        if value_col in run_values:
            fig.add_vline(x=run_values[value_col],line_width=3,line_color="#f2f4f7",
                          annotation_text=f"Run: {run_values[value_col]:.1f}",annotation_position="bottom right",annotation_font_color="#f2f4f7")
        fig.update_layout(
            height=430,margin=dict(l=20,r=20,t=25,b=20),
            paper_bgcolor="#12161c",plot_bgcolor="#12161c",font=dict(color="#f2f4f7"),
            xaxis=dict(title=selected_label,gridcolor="#2a3038",zeroline=False),
            yaxis=dict(title="Frequency",gridcolor="#2a3038",zeroline=False),
            bargap=0.05,showlegend=False,
        )
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

        pct_df=pd.DataFrame({"Percentile":list(percentiles.keys()),"Value":list(percentiles.values())})
        export_cols=[c for c in ["simulation","daily_demand",value_col] if c in results.columns]
        selected_export=results[export_cols].rename(columns={value_col:selected_label})
        d1,d2,d3=st.columns(3)
        d1.download_button("Download selected distribution",selected_export.to_csv(index=False).encode("utf-8"),"jeddah_monte_carlo_selected.csv","text/csv",key="download_mc_selected")
        d2.download_button("Download all gate Monte Carlo results",results.to_csv(index=False).encode("utf-8"),"jeddah_monte_carlo_all_gate_results.csv","text/csv",key="download_mc_all")
        d3.download_button("Download percentile summary",pct_df.to_csv(index=False).encode("utf-8"),"jeddah_monte_carlo_gate_percentiles.csv","text/csv",key="download_mc_percentiles")
    else:
        st.info(f"No Monte Carlo results are available for {selected_label}.")

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
