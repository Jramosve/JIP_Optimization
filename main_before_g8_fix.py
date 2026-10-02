# ============================================================
# JEDDAH ISLAMIC PORT - TRUCK TRAFFIC MODEL
# V1.1
# ============================================================

import math
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import simpy

try:
    from shapely.geometry import LineString, Point, MultiPoint
    from shapely.ops import unary_union, snap, split
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# ============================================================
# 1. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# DEMAND
# ------------------------------------------------------------

TRUCK_MOVEMENTS_PER_DAY = 10_500
INTERNAL_MOVEMENTS_SHARE = 0.05
INTERNAL_MOVEMENTS_PER_DAY = int(round(TRUCK_MOVEMENTS_PER_DAY * INTERNAL_MOVEMENTS_SHARE))

RANDOM_SEED = 42

# ------------------------------------------------------------
# ROUTE / FLOW MANAGEMENT
# ------------------------------------------------------------
# 0.00 = always shortest free-flow path.
# 1.00 = use an alternative path whenever available.
ROUTE_RANDOMIZATION = 0.00
ROUTE_ALTERNATIVE_COUNT = 3

# Optional explicit route overrides keyed by (origin, destination).
# When empty, the model uses shortest free-flow path routing.
ROUTE_OVERRIDES = {}

# Gate queue sampling interval used by the dashboard/output.
QUEUE_SNAPSHOT_INTERVAL_MIN = 5.0

# ------------------------------------------------------------
# ARRIVAL TIME VARIABILITY
# ------------------------------------------------------------
# 0.00 = use the defined hourly profile exactly.
# 0.10 = low/moderate daily variation around the profile.
# 0.15 = recommended starting point.
# 0.25 = high variation.
#
# This changes the number of trucks assigned to each hour while
# preserving the total daily truck demand exactly.
ARRIVAL_TIME_RANDOMIZATION = 0.15

# ------------------------------------------------------------
# GATE APPOINTMENT / FLOW MANAGEMENT
# ------------------------------------------------------------
# When enabled, port-entry appointments are capped at a maximum
# number of trucks per hour. Demand that cannot be released in a
# given hour becomes uncovered demand and rolls into the next hour.
APPOINTMENT_MANAGEMENT_ENABLED = True
MAX_PORT_ENTRIES_PER_HOUR = 500



# ------------------------------------------------------------
# TRAFFIC
# ------------------------------------------------------------

AVERAGE_SPEED_KMH = 30.0


# ------------------------------------------------------------
# ROAD CAPACITY
# IMPORTANT:
# These are assumptions and should later be replaced with
# actual lane counts / capacities.
# ------------------------------------------------------------

LANES_PER_DIRECTION = 2
CAPACITY_PER_LANE_VPH = 800

ROAD_CAPACITY_VPH = (
    LANES_PER_DIRECTION
    * CAPACITY_PER_LANE_VPH
)


# ------------------------------------------------------------
# JUNCTIONS / INTERSECTIONS
# IMPORTANT:
# Screening assumptions. These represent effective uncontrolled
# junction capacity plus an additional maneuver delay.
# Replace with observed/calibrated values when available.
# ------------------------------------------------------------

JUNCTIONS = {
    "N01": {"control": "UNCONTROLLED", "capacity_vph": 700, "maneuver_delay_sec": 8.0},
    "N02": {"control": "UNCONTROLLED", "capacity_vph": 700, "maneuver_delay_sec": 6.0},
    "N03": {"control": "UNCONTROLLED", "capacity_vph": 700, "maneuver_delay_sec": 8.0},
}


# ------------------------------------------------------------
# GATE LANES
# IMPORTANT:
# These are still assumptions.
# ------------------------------------------------------------

GATE_LANES = {
    "G1": {"entry": 2, "exit": 1},
    "G4": {"entry": 3, "exit": 0},
    "G6": {"entry": 0, "exit": 0},
    "G8": {"entry": 0, "exit": 3},
    "G9": {"entry": 3, "exit": 2},
}


# Physical queue storage at the exit-gate approach.
# This is kept separate from the upstream road-link storage: the gate queue
# must be allowed to build before spillback propagates into the network.
# These are screening assumptions and should be replaced by measured queue-lane
# storage when available.
EXIT_GATE_QUEUE_STORAGE_TRUCKS = {
    "G1": 120,
    "G9": 250,
}


# ------------------------------------------------------------
# GATE PROCESSING TIMES
# seconds / truck
# ------------------------------------------------------------

GATE_TIMES_SEC = {

    # User-provided assumption.
    "FULL_ENTRY": 35.0,

    "FULL_EXIT": 30.0,

    "EMPTY_ENTRY": 20.0,

    "EMPTY_EXIT": 20.0,
}


# ------------------------------------------------------------
# TERMINAL PROCESSING TIMES
# minutes
# ------------------------------------------------------------

TERMINAL_PROCESS_MIN = {
    "RSGT": {"FULL": 75.0, "EMPTY": 75.0},
    "DPW": {"FULL": 60.0, "EMPTY": 60.0},
    "MPT": {"FULL": 75.0, "EMPTY": 75.0},
    "LOGISTICS": {"FULL": 75.0, "EMPTY": 75.0},
}


# ------------------------------------------------------------
# INTERNAL TERMINAL GATE CROSSING
# seconds
# ------------------------------------------------------------

INTERNAL_GATE_CROSSING_SEC = {

    "RSGT": 8.0,

    "DPW": 8.0,

    "MPT": 0.0,

    "LOGISTICS": 0.0,
}


# ------------------------------------------------------------
# TERMINAL TRAFFIC SHARES
#
# Illustrative assumptions.
# Replace with actual data when available.
# ------------------------------------------------------------

CARGOS = ["FULL", "EMPTY"]


TERMINAL_SHARES = {
    "RSGT": 0.40,
    "DPW": 0.35,
    "MPT": 0.20,
    "LOGISTICS": 0.05,
}

# ------------------------------------------------------------
# CARGO COMPOSITION
# ------------------------------------------------------------

CARGO_SHARES = {
    "RSGT": {"FULL": 0.70, "EMPTY": 0.30},
    "DPW": {"FULL": 0.70, "EMPTY": 0.30},
    "MPT": {"FULL": 0.70, "EMPTY": 0.30},
    "LOGISTICS": {"FULL": 0.70, "EMPTY": 0.30},
}


# Gate processing times are defined only by cargo type and operation.
# They are independent of terminal.

# Terminal + cargo access-gate allocation. Shares must sum to 100%
# for each terminal/cargo combination.
ENTRY_ACCESS_RULES = {
    ("RSGT", "FULL"): [("G4", 1.00)],
    ("RSGT", "EMPTY"): [("G9", 1.00)],
    ("DPW", "FULL"): [("G4", 1.00)],
    ("DPW", "EMPTY"): [("G9", 1.00)],
    ("MPT", "FULL"): [("G4", 1.00)],
    ("MPT", "EMPTY"): [("G9", 1.00)],
    ("LOGISTICS", "FULL"): [("G1", 0.20), ("G9", 0.80)],
    ("LOGISTICS", "EMPTY"): [("G1", 0.20), ("G9", 0.80)],
}

# The N2-N1 corridor follows the named KML path
# N2-N7-N6-N5-N4-N8-N1. It currently has 3 lanes per direction and is
# designed to be switchable to 4 lanes for future testing.
N2_N1_CORRIDOR_LANES = 3


# ------------------------------------------------------------
# MPT DESTINATION SPLIT
# ------------------------------------------------------------

MPT_DESTINATION_SPLIT = {

    "MPT1": 0.40,

    "MPT2": 0.35,

    "MPT3": 0.25,
}


# Dummy daily demand for Logistics-area destinations.
LOGISTICS_DESTINATION_SHARE = {
    "Logipoint": 0.50,
    "CMA CGM": 0.25,
    "Bahri Logistics": 0.25,
}

# Backward-compatible display/output values. Actual logistics demand is always
# derived as 5% of total daily truck movements.
LOGISTICS_DESTINATION_DAILY = {
    "Logipoint": 0,
    "CMA CGM": 0,
    "Bahri Logistics": 0,
}


# ------------------------------------------------------------
# TEMPORARY LOGISTICS CONNECTION
#
# There is currently no dedicated Logistics road in the KML.
# We therefore connect Logistics to N01 temporarily.
# ------------------------------------------------------------

LOGISTICS_NETWORK_NODE = "Logipoint"


# ============================================================
# 2. GATE ROUTING
# ============================================================

ENTRY_GATE_RULES = {

    # RSGT
    ("RSGT", "FULL"): [
        ("G4", 1.00)
    ],

    ("RSGT", "EMPTY"): [
        ("G9", 1.00)
    ],


    # DPW
    ("DPW", "FULL"): [
        ("G4", 1.00)
    ],

    ("DPW", "EMPTY"): [
        ("G9", 1.00)
    ],


    # MPT
    ("MPT", "FULL"): [
        ("G4", 1.00)
    ],

    ("MPT", "EMPTY"): [
        ("G9", 1.00)
    ],


    # LOGISTICS
    ("LOGISTICS", "FULL"): [("G1", 0.20), ("G9", 0.80)],
    ("LOGISTICS", "EMPTY"): [("G1", 0.20), ("G9", 0.80)],
}


EXIT_GATE_RULES = {

    # RSGT
    ("RSGT", "FULL"): [
        ("G8", 0.80),
        ("G1", 0.20)
    ],

    ("RSGT", "EMPTY"): [
        ("G9", 0.80),
        ("G1", 0.20)
    ],


    # DPW
    ("DPW", "FULL"): [
        ("G8", 1.00)
    ],

    ("DPW", "EMPTY"): [
        ("G9", 1.00)
    ],


    # MPT
    ("MPT", "FULL"): [
        ("G8", 1.00)
    ],

    ("MPT", "EMPTY"): [
        ("G8", 1.00)
    ],


    # LOGISTICS
    ("LOGISTICS", "FULL"): [
        ("G8", 0.70),
        ("G1", 0.30)
    ],

    ("LOGISTICS", "EMPTY"): [
        ("G1", 0.20),
        ("G9", 0.80)
    ],
}


# ============================================================
# 3. FILES
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


KML_FILE = (
    BASE_DIR
    / "jip_routes.kml"
)


TRUCK_OUTPUT = (
    BASE_DIR
    / "simulation_trucks_v060.csv"
)


GATE_OUTPUT = (
    BASE_DIR
    / "simulation_gates_v060.csv"
)


ROAD_OUTPUT = (
    BASE_DIR
    / "simulation_roads_v060.csv"
)


KML_OUTPUT = (
    BASE_DIR
    / "simulation_roads_v060.kml"
)


JUNCTION_OUTPUT = (
    BASE_DIR
    / "simulation_junctions_v060.csv"
)


HOURLY_KML_OUTPUT = (
    BASE_DIR
    / "simulation_congestion_hourly_v060.kml"
)


# ============================================================
# 4. BASIC FUNCTIONS
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0088

    phi1 = math.radians(lat1)

    phi2 = math.radians(lat2)

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dphi / 2) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return (
        2
        * R
        * math.asin(
            math.sqrt(a)
        )
    )


def route_length_km(
    coordinates
):

    total = 0.0

    for i in range(
        len(coordinates) - 1
    ):

        lon1, lat1 = coordinates[i]

        lon2, lat2 = coordinates[i + 1]

        total += haversine_km(
            lat1,
            lon1,
            lat2,
            lon2
        )

    return total


def travel_time_min(
    distance_km
):

    return (
        distance_km
        / AVERAGE_SPEED_KMH
        * 60
    )


def weighted_choice(items, weights, rng):
    if not items:
        raise ValueError("No choices available.")
    if len(items) != len(weights):
        raise ValueError("Items and weights must have the same length.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Choice weights must sum to a positive value.")
    return rng.choices(items, weights=weights, k=1)[0]


def choose_flow_option(options, rng):
    cleaned = [(node, float(share)) for node, share in options if float(share) > 0]
    if not cleaned:
        raise ValueError("Flow rule has no positive shares.")
    total = sum(share for _, share in cleaned)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Flow shares must sum to 1.0; got {total:.6f}")
    return weighted_choice(
        [node for node, _ in cleaned],
        [share for _, share in cleaned],
        rng,
    )


def choose_network_path(graph, start, destination, rng, route_overrides=None):
    """Return an explicit configured path when provided, otherwise shortest path."""
    overrides = ROUTE_OVERRIDES if route_overrides is None else route_overrides
    override = overrides.get((start, destination)) if overrides else None

    if override:
        path = list(override)
        if path[0] != start or path[-1] != destination:
            raise ValueError(
                f"Route override {start} -> {destination} must start at {start} "
                f"and finish at {destination}."
            )
        missing_links = [
            f"{a} -> {b}" for a, b in zip(path[:-1], path[1:])
            if not graph.has_edge(a, b)
        ]
        if missing_links:
            raise ValueError(
                f"Invalid route override {start} -> {destination}; "
                f"missing links: {', '.join(missing_links)}"
            )
        return path

    try:
        generator = nx.shortest_simple_paths(
            graph, source=start, target=destination, weight="travel_time_min"
        )
        candidates = []
        for path in generator:
            candidates.append(path)
            if len(candidates) >= max(1, ROUTE_ALTERNATIVE_COUNT):
                break
    except nx.NetworkXNoPath:
        raise RuntimeError(f"No route found from {start} to {destination}")

    if len(candidates) == 1 or ROUTE_RANDOMIZATION <= 0:
        return candidates[0]

    if rng.random() >= ROUTE_RANDOMIZATION:
        return candidates[0]

    costs = [
        sum(graph[a][b]["travel_time_min"] for a, b in zip(path[:-1], path[1:]))
        for path in candidates
    ]
    min_cost = min(costs)
    weights = [max(0.05, min_cost / c) for c in costs]
    return weighted_choice(candidates, weights, rng)


def validate_route_overrides(graph, route_overrides):
    """Validate explicit OD paths before the simulation starts."""
    errors = []
    for (start, destination), path in (route_overrides or {}).items():
        if not path or path[0] != start or path[-1] != destination:
            errors.append(f"{start} -> {destination}: invalid endpoints")
            continue
        for a, b in zip(path[:-1], path[1:]):
            if not graph.has_edge(a, b):
                errors.append(f"{start} -> {destination}: missing link {a} -> {b}")
    return errors


# ============================================================
# 5. KML READER
# ============================================================

def normalize_node_name(name):

    name = name.strip()

    # -----------------------------------------
    # Remove "Route " prefix if present
    # -----------------------------------------

    if name.startswith("Route "):
        name = name[6:]


    # -----------------------------------------
    # Basic cleanup
    # -----------------------------------------

    name = re.sub(r"\s+", " ", name).strip()


    # -----------------------------------------
    # Google Earth gate names
    # -----------------------------------------

    if name == "RSGT Gate":
        return "RSGT"

    if name == "DPW Gate":
        return "DPW"


    # -----------------------------------------
    # MPT names
    # -----------------------------------------

    name = name.replace("MPT 1", "MPT1")
    name = name.replace("MPT 2", "MPT2")
    name = name.replace("MPT 3", "MPT3")


    # -----------------------------------------
    # Node names
    #
    # Google Earth:
    #   Node 1
    #   Node 2
    #   Node 3
    #
    # Internal model:
    #   N01
    #   N02
    #   N03
    # -----------------------------------------

    node_mapping = {
        "Node 1": "N01",
        "Node 2": "N02",
        "Node 3": "N03",
        "Node 4": "N04",
        "Node 5": "N05",
        "Node 6": "N06",
        "Node 7": "N07",

        "Node1": "N01",
        "Node2": "N02",
        "Node3": "N03",
        "Node4": "N04",
        "Node5": "N05",
        "Node6": "N06",
        "Node7": "N07",

        "N1": "N01",
        "N2": "N02",
        "N3": "N03",
        "N4": "N04",
        "N5": "N05",
        "N6": "N06",
        "N7": "N07",

        "N01": "N01",
        "N02": "N02",
        "N03": "N03",
        "N04": "N04",
        "N05": "N05",
        "N06": "N06",
        "N07": "N07",
        "Node 8": "N08",
        "Node 9": "N09",
        "Node 10": "N10",
        "Node8": "N08",
        "Node9": "N09",
        "Node10": "N10",
        "N8": "N08",
        "N9": "N09",
        "N10": "N10",
        "N08": "N08",
        "N09": "N09",
        "N10": "N10",
    }

    if name == "Bahri":
        return "Bahri Logistics"
    if name in {"CMA", "CMA CGM"}:
        return "CMA CGM"

    if name in node_mapping:
        return node_mapping[name]


    # -----------------------------------------
    # Return original name
    # -----------------------------------------

    return name


def read_kml_routes():

    if not KML_FILE.exists():

        raise FileNotFoundError(
            "\n\nKML file not found:\n"
            f"{KML_FILE}\n\n"
            "Make sure your Google Earth file is saved "
            "as 'jip_routes.kml' in the same folder as main.py."
        )


    raw = KML_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    )


    # Google Earth may add <?earth ...?> after </kml>.
    # We only parse the actual XML section.

    end_marker = "</kml>"

    end_position = (
        raw.lower()
        .find(end_marker)
    )

    if end_position == -1:

        raise RuntimeError(
            "Could not find </kml> in the KML file."
        )


    xml_text = raw[
        :end_position
        + len(end_marker)
    ]


    root = ET.fromstring(
        xml_text
    )


    namespace = {
        "kml":
        "http://www.opengis.net/kml/2.2"
    }


    # Explicit Google Earth Point placemarks are the authoritative node
    # coordinates. This prevents N04/N05 (and similar nodes) from being
    # misplaced when a manually drawn LineString happens to be stored in
    # the opposite direction to its name.
    node_points = {}
    for placemark in root.findall(".//kml:Placemark", namespace):
        name_element = placemark.find("kml:name", namespace)
        coordinates_element = placemark.find(".//kml:Point/kml:coordinates", namespace)
        if name_element is None or coordinates_element is None:
            continue
        name = normalize_node_name(name_element.text or "")
        values = (coordinates_element.text or "").strip().split(",")
        if len(values) >= 2:
            node_points[name] = (float(values[0]), float(values[1]))

    routes = []


    for placemark in root.findall(
        ".//kml:Placemark",
        namespace
    ):

        name_element = (
            placemark.find(
                "kml:name",
                namespace
            )
        )


        coordinates_element = (
            placemark.find(
                ".//kml:LineString/"
                "kml:coordinates",
                namespace
            )
        )


        if (
            name_element is None
            or coordinates_element is None
        ):

            continue


        name = (
            name_element.text
            .strip()
        )


        parts = [
            normalize_node_name(
                x
            )
            for x in name.split(
                " - "
            )
        ]


        if len(parts) != 2:

            continue


        start_node = parts[0]

        end_node = parts[1]


        coordinates = []


        raw_coordinates = (
            coordinates_element.text
            or ""
        )


        for item in raw_coordinates.split():

            values = item.split(",")


            if len(values) < 2:

                continue


            lon = float(
                values[0]
            )

            lat = float(
                values[1]
            )


            coordinates.append(
                (
                    lon,
                    lat
                )
            )


        if len(coordinates) < 2:
            continue

        # Google Earth may store the geometry in reverse order relative to
        # the route name. Re-orient it using the named Point placemarks and
        # snap both endpoints exactly onto those points.
        start_point = node_points.get(start_node)
        end_point = node_points.get(end_node)
        if start_point is not None and end_point is not None:
            forward_error = haversine_km(
                coordinates[0][1], coordinates[0][0],
                start_point[1], start_point[0]
            ) + haversine_km(
                coordinates[-1][1], coordinates[-1][0],
                end_point[1], end_point[0]
            )
            reverse_error = haversine_km(
                coordinates[-1][1], coordinates[-1][0],
                start_point[1], start_point[0]
            ) + haversine_km(
                coordinates[0][1], coordinates[0][0],
                end_point[1], end_point[0]
            )
            if reverse_error < forward_error:
                coordinates.reverse()
            coordinates[0] = start_point
            coordinates[-1] = end_point

        distance = route_length_km(
            coordinates
        )


        time_min = travel_time_min(
            distance
        )


        routes.append(
            {
                "name": name,
                "start": start_node,
                "end": end_node,
                "coordinates": coordinates,
                "distance_km": distance,
                "travel_time_min": time_min
            }
        )


    return routes


# ============================================================
# 6. BUILD ROAD NETWORK
# ============================================================

def _project_lonlat(lon, lat, lat0):
    meters_per_degree = 111320.0
    x = lon * meters_per_degree * math.cos(math.radians(lat0))
    y = lat * meters_per_degree
    return x, y


def _unproject_xy(x, y, lat0):
    meters_per_degree = 111320.0
    lon = x / (meters_per_degree * math.cos(math.radians(lat0)))
    lat = y / meters_per_degree
    return lon, lat


def read_kml_point_nodes():
    """Read all explicit Google Earth Point placemarks, including isolated points."""
    if not KML_FILE.exists():
        return {}
    raw = KML_FILE.read_text(encoding="utf-8", errors="ignore")
    end_marker = "</kml>"
    end_position = raw.lower().find(end_marker)
    if end_position == -1:
        return {}
    root = ET.fromstring(raw[:end_position + len(end_marker)])
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    points = {}
    for placemark in root.findall(".//kml:Placemark", namespace):
        name_element = placemark.find("kml:name", namespace)
        coord = placemark.find(".//kml:Point/kml:coordinates", namespace)
        if name_element is None or coord is None:
            continue
        values = (coord.text or "").strip().split(",")
        if len(values) >= 2:
            points[normalize_node_name(name_element.text or "")] = (float(values[0]), float(values[1]))
    return points


def build_network(routes):
    """Build a clean screening network directly from the named KML corridors.

    The KML contains 10 manually digitised physical corridors. For the
    screening model we deliberately keep only the named endpoints from the
    KML and do NOT convert every polyline vertex/crossing into an intersection.
    This avoids creating artificial INTxxx junctions.

    A later GIS-centreline version can split the shared road sections properly
    once we have a more precise road network. For now each KML corridor is a
    physical road with two independent directions.
    """
    if not routes:
        raise RuntimeError("No KML routes were found.")

    graph = nx.DiGraph()

    # Add every explicit Google Earth Point, including points that are not yet
    # connected by a road (e.g. N03), so the dashboard map shows the complete
    # reference network.
    for node, (lon, lat) in read_kml_point_nodes().items():
        graph.add_node(node, lon=float(lon), lat=float(lat), point_only=True)

    # Coordinates for named route nodes come from the KML route endpoints,
    # while explicit Point placemarks remain authoritative when available.
    node_points = {}
    for route in routes:
        node_points.setdefault(route["start"], []).append(route["coordinates"][0])
        node_points.setdefault(route["end"], []).append(route["coordinates"][-1])

    point_nodes = read_kml_point_nodes()
    for node, pts in node_points.items():
        if node in point_nodes:
            lon, lat = point_nodes[node]
        else:
            lon = float(np.mean([p[0] for p in pts]))
            lat = float(np.mean([p[1] for p in pts]))
        graph.add_node(node, lon=float(lon), lat=float(lat), point_only=False)

    for idx, route in enumerate(routes, start=1):
        a = route["start"]
        b = route["end"]
        geometry = list(route["coordinates"])
        road_id = f"KML_ROAD_{idx:02d}"

        # Special N2-N1 corridor: N02-N07-N06-N05-N04-N08-N01.
        # Apply the configurable 3/4 lane capacity to each physical segment
        # that forms the N2-N1 corridor. Other roads use the default lane count.
        corridor_pairs = {
            frozenset(("N01", "N08")),
            frozenset(("N08", "N04")),
            frozenset(("N04", "N05")),
            frozenset(("N05", "N06")),
            frozenset(("N06", "N07")),
            frozenset(("N07", "N02")),
        }
        is_n2_n1 = frozenset((a, b)) in corridor_pairs
        edge_lanes = N2_N1_CORRIDOR_LANES if is_n2_n1 else LANES_PER_DIRECTION
        attrs = {
            "road_id": road_id,
            "physical_road": road_id,
            "distance_km": float(route["distance_km"]),
            "travel_time_min": float(route["travel_time_min"]),
            "capacity_vph": edge_lanes * CAPACITY_PER_LANE_VPH,
            "lanes": edge_lanes,
            "corridor": "N2-N1" if is_n2_n1 else "DEFAULT",
            "geometry": geometry,
            "network_source": "KML_CORRIDOR",
            "kml_name": route["name"],
        }

        # One physical road, two independent directional resources.
        graph.add_edge(a, b, **attrs, direction="AB")
        reverse_attrs = dict(attrs)
        reverse_attrs["geometry"] = list(reversed(geometry))
        graph.add_edge(
            b,
            a,
            **reverse_attrs,
            direction="BA",
        )

    return graph


# ============================================================
# 7. DESTINATION NODE MAPPING
# ============================================================

def get_destination_node(
    terminal,
    mpt_destination=None
):
    """Map traffic to the physical destination node in the KML."""
    if terminal == "MPT":
        if mpt_destination is None:
            raise ValueError("MPT truck has no mpt_destination assigned.")
        return mpt_destination
    if terminal == "LOGISTICS":
        return mpt_destination or LOGISTICS_NETWORK_NODE
    if terminal in {"RSGT", "DPW"}:
        return terminal
    raise ValueError(f"Unknown terminal '{terminal}'. Cannot determine network destination.")


# ============================================================
# 8. DEMAND PROFILE
# ============================================================
# 8. DEMAND PROFILE
# ============================================================
# 8. DEMAND PROFILE
# ============================================================

def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def create_base_hourly_profile():
    """Observed 24-hour truck distribution supplied for JIP.

    Values are hourly shares of total daily traffic and sum to 100%.
    Trucks are later distributed approximately uniformly within each hour.
    """
    hourly_shares = np.array([
        0.035, 0.030, 0.021, 0.015, 0.018, 0.019,
        0.036, 0.036, 0.043, 0.048, 0.048, 0.051,
        0.050, 0.048, 0.055, 0.056, 0.053, 0.056,
        0.048, 0.046, 0.047, 0.051, 0.048, 0.042,
    ], dtype=float)
    hourly_shares /= hourly_shares.sum()
    return np.repeat(hourly_shares / 60.0, 60)


def create_hourly_profile(rng=None, variability=None):
    """Return a 24-hour aggregation of the smooth 1-minute demand profile."""
    minute_profile = create_base_hourly_profile()
    if variability is None:
        variability = ARRIVAL_TIME_RANDOMIZATION
    variability = float(variability)
    if not 0 <= variability <= 1:
        raise ValueError("ARRIVAL_TIME_RANDOMIZATION must be between 0 and 1.")
    if variability == 0:
        return np.array([minute_profile[h * 60:(h + 1) * 60].sum() for h in range(24)])

    if rng is None:
        rng = random.Random(RANDOM_SEED)
    # Apply small smooth daily variation to hourly totals, then distribute it
    # back across the minute-level profile.
    hourly = np.array([minute_profile[h * 60:(h + 1) * 60].sum() for h in range(24)])
    noise = np.array([math.exp(rng.gauss(0.0, variability * 0.12)) for _ in range(24)])
    hourly *= noise
    hourly /= hourly.sum()
    return hourly


def generate_arrival_times(number_of_trucks, rng, variability=None):
    """Generate exact demand using the observed hourly profile.

    The base profile is fixed at the observed 24-hour shares. Variability is
    introduced only as a controlled Monte Carlo-style perturbation of hourly
    shares; within each hour trucks are distributed approximately uniformly.
    """
    hourly = np.array([create_base_hourly_profile()[h * 60:(h + 1) * 60].sum() for h in range(24)], dtype=float)
    variability = ARRIVAL_TIME_RANDOMIZATION if variability is None else float(variability)
    if not 0 <= variability <= 1:
        raise ValueError("ARRIVAL_TIME_RANDOMIZATION must be between 0 and 1.")
    if variability > 0:
        noise = np.exp(np.array([rng.gauss(0.0, variability * 0.12) for _ in range(24)]))
        hourly *= noise
        hourly /= hourly.sum()
    # Allocate an exact number of trucks to each hour, preserving the daily total.
    raw = hourly * int(number_of_trucks)
    counts = np.floor(raw).astype(int)
    remainder = int(number_of_trucks) - int(counts.sum())
    if remainder > 0:
        for idx in np.argsort(-(raw - counts))[:remainder]:
            counts[idx] += 1
    arrivals = []
    for h, count in enumerate(counts):
        if count <= 0:
            continue
        # Approx. uniform through the hour with a small deterministic jitter.
        for i in range(int(count)):
            frac = (i + 0.5) / count
            jitter = rng.uniform(-0.25 / max(count, 1), 0.25 / max(count, 1))
            arrivals.append(h * 60.0 + min(59.999, max(0.0, (frac + jitter) * 60.0)))
    arrivals.sort()
    return arrivals


# ============================================================
# 9. TRUCK GENERATION
# ============================================================

def choose_terminal(
    rng
):

    terminals = list(
        TERMINAL_SHARES.keys()
    )

    weights = list(
        TERMINAL_SHARES.values()
    )

    return weighted_choice(
        terminals,
        weights,
        rng
    )


def choose_cargo(
    terminal,
    rng
):

    cargos = list(
        CARGO_SHARES[
            terminal
        ].keys()
    )

    weights = list(
        CARGO_SHARES[
            terminal
        ].values()
    )

    return weighted_choice(
        cargos,
        weights,
        rng
    )


def choose_gate(rules, terminal, cargo, rng):
    return choose_flow_option(rules[(terminal, cargo)], rng)


def choose_mpt_destination(
    rng
):

    destinations = list(
        MPT_DESTINATION_SPLIT.keys()
    )

    weights = list(
        MPT_DESTINATION_SPLIT.values()
    )

    return weighted_choice(
        destinations,
        weights,
        rng
    )

# ============================================================
# DESTINATION NODE MAPPING
# ============================================================

def get_destination_node(
    terminal,
    mpt_destination=None
):
    """
    Map each terminal to the corresponding physical node in the
    JIP road network defined in the V1.1 KML.

    MPT is split between MPT1, MPT2 and MPT3.
    Logistics is represented by the Logipoint node.
    """

    if terminal == "MPT":

        if mpt_destination is None:
            raise ValueError(
                "MPT truck has no mpt_destination assigned."
            )

        return mpt_destination

    if terminal == "LOGISTICS":
        return LOGISTICS_NETWORK_NODE

    if terminal in {
        "RSGT",
        "DPW"
    }:
        return terminal

    raise ValueError(
        f"Unknown terminal '{terminal}'. "
        "Cannot determine network destination."
    )
def apply_entry_appointment_cap(arrival_times, max_entries_per_hour):
    """Apply a port-wide hourly entry appointment cap.

    Scheduled demand is preserved exactly. If demand in an hour exceeds the
    appointment cap, the excess becomes uncovered demand and is carried FIFO
    into the following hour. The process continues beyond 24:00 if necessary,
    so no trucks are silently dropped from the simulation.
    """
    if max_entries_per_hour is None:
        return sorted(arrival_times), pd.DataFrame()

    cap = int(max_entries_per_hour)
    if cap <= 0:
        raise ValueError("max_entries_per_hour must be > 0")

    # Keep the original scheduled demand by hour.
    by_hour = {}
    for t in arrival_times:
        h = max(0, int(float(t) // 60))
        by_hour.setdefault(h, []).append(float(t))
    for h in by_hour:
        by_hour[h].sort()

    backlog = []
    released_times = []
    rows = []
    hour = 0

    # Continue until every scheduled truck has received an appointment.
    # This can extend beyond 24:00 when the hourly cap is restrictive.
    while hour < 24 or backlog or hour in by_hour:
        current = by_hour.get(hour, [])
        pool = backlog + current
        demand_available = len(pool)
        release = min(cap, demand_available)
        release_items = pool[:release]
        backlog = pool[release:]

        # Spread released appointments continuously through the hour rather
        # than putting them all at the exact start of the hour.
        if release:
            for i in range(release):
                minute = (i + 0.5) / release * 60.0
                released_times.append(hour * 60.0 + minute)

        rows.append({
            "hour": hour,
            "scheduled_demand": len(current),
            "uncovered_from_previous": len(pool) - len(current),
            "appointment_capacity": cap,
            "released_entries": release,
            "uncovered_end_of_hour": len(backlog),
        })
        hour += 1

    # Sanity check: appointment management must never silently lose demand.
    if len(released_times) != len(arrival_times):
        raise RuntimeError(
            f"Appointment management released {len(released_times):,} "
            f"of {len(arrival_times):,} scheduled trucks."
        )

    return released_times, pd.DataFrame(rows)

def allocate_integer_counts(total, weights):
    keys=list(weights.keys()); raw=np.array([float(weights[k]) for k in keys]); raw=raw/raw.sum()*int(total)
    base=np.floor(raw).astype(int); remainder=int(total)-int(base.sum())
    for idx in np.argsort(-(raw-base))[:remainder]: base[idx]+=1
    return {k:int(v) for k,v in zip(keys,base)}

def build_daily_terminal_destinations(total_trucks, logistics_destination_daily=None):
    """Build terminal assignments directly from editable terminal shares."""
    total = int(total_trucks)
    counts = allocate_integer_counts(total, TERMINAL_SHARES)
    assignments = []
    for terminal, count in counts.items():
        if terminal != "LOGISTICS":
            assignments.extend([(terminal, None)] * int(count))
        else:
            destinations = list(LOGISTICS_DESTINATION_SHARE.keys())
            weights = LOGISTICS_DESTINATION_SHARE
            dcounts = allocate_integer_counts(int(count), weights)
            for d, dc in dcounts.items():
                assignments.extend([("LOGISTICS", d)] * int(dc))
    return assignments


def create_trucks(
    arrival_time_variability=None, appointment_management_enabled=None,
    max_entries_per_hour=None, logistics_destination_daily=None,
):
    rng=random.Random(RANDOM_SEED)
    assignments=build_daily_terminal_destinations(TRUCK_MOVEMENTS_PER_DAY, logistics_destination_daily)
    rng.shuffle(assignments)
    arrival_times=generate_arrival_times(TRUCK_MOVEMENTS_PER_DAY,rng,variability=arrival_time_variability)
    if appointment_management_enabled is None: appointment_management_enabled=APPOINTMENT_MANAGEMENT_ENABLED
    if max_entries_per_hour is None: max_entries_per_hour=MAX_PORT_ENTRIES_PER_HOUR
    appointment_df=pd.DataFrame()
    if appointment_management_enabled: arrival_times,appointment_df=apply_entry_appointment_cap(arrival_times,max_entries_per_hour)
    trucks=[]
    for i in range(TRUCK_MOVEMENTS_PER_DAY):
        terminal,destination=assignments[i]; cargo=choose_cargo(terminal,rng)
        entry_gate=choose_flow_option(ENTRY_ACCESS_RULES[(terminal, cargo)], rng); exit_gate=choose_gate(EXIT_GATE_RULES,terminal,cargo,rng)
        if terminal=="MPT": network_destination=choose_mpt_destination(rng)
        elif terminal=="LOGISTICS": network_destination=destination
        else: network_destination=None
        trucks.append({"truck_id":f"T{i+1:05d}","arrival_min":arrival_times[i],"terminal":terminal,"cargo":cargo,"entry_gate":entry_gate,"exit_gate":exit_gate,"mpt_destination":network_destination,"network_destination":get_destination_node(terminal,network_destination)})
    return trucks,appointment_df


# ============================================================
# 10. ROAD MANAGER
# ============================================================
# 10. ROAD MANAGER
# ============================================================

class RoadManager:

    def __init__(
        self,
        env,
        graph
    ):

        self.env = env

        self.graph = graph

        self.resources = {}
        self.spillback_resources = {}
        self.spillback_capacity = {}

        self.traversals = []
        self.spillback_activities = []


        for a, b, data in (
            graph.edges(
                data=True
            )
        ):

            travel_min = (
                data[
                    "travel_time_min"
                ]
            )


            capacity_vph = (
                data[
                    "capacity_vph"
                ]
            )


            # Approximate storage capacity of the link.

            simultaneous_capacity = max(

                1,

                int(
                    round(
                        capacity_vph
                        * travel_min
                        / 60
                    )
                )
            )


            self.resources[(a, b)] = simpy.Resource(env, capacity=simultaneous_capacity)

            # Keep gate queue storage separate from generic link storage.
            # Otherwise the gate queue gets artificially capped by
            # capacity_vph * travel_time, which is a throughput/storage proxy
            # for the road link and is NOT the physical gate queue capacity.
            # For the three exit-gate approaches, use the explicit gate queue
            # storage assumption; other links retain their normal link storage.
            gate_for_approach = b if b in EXIT_GATE_QUEUE_STORAGE_TRUCKS else None
            if gate_for_approach is not None:
                spill_storage = int(EXIT_GATE_QUEUE_STORAGE_TRUCKS[gate_for_approach])
            else:
                spill_storage = simultaneous_capacity

            self.spillback_capacity[(a, b)] = spill_storage
            self.spillback_resources[(a, b)] = simpy.Resource(env, capacity=spill_storage)


    def traverse(
        self,
        truck_id,
        a,
        b
    ):

        data = self.graph[
            a
        ][
            b
        ]


        resource = self.resources[
            (
                a,
                b
            )
        ]


        requested_at = (
            self.env.now
        )


        with resource.request() as request:

            yield request


            service_start = (
                self.env.now
            )


            wait_min = (
                service_start
                - requested_at
            )


            travel_min = (
                data[
                    "travel_time_min"
                ]
            )


            yield self.env.timeout(
                travel_min
            )


            finished_at = (
                self.env.now
            )


        self.traversals.append(

            {

                "truck_id":
                    truck_id,

                "from_node":
                    a,

                "to_node":
                    b,

                "road_id":
                    data[
                        "road_id"
                    ],

                "physical_road":
                    data[
                        "physical_road"
                    ],

                "distance_km":
                    data[
                        "distance_km"
                    ],

                "capacity_vph":
                    data[
                        "capacity_vph"
                    ],

                "lanes":
                    data[
                        "lanes"
                    ],

                "requested_min":
                    requested_at,

                "start_min":
                    service_start,

                "finish_min":
                    finished_at,

                "wait_min":
                    wait_min,

                "travel_min":
                    travel_min
            }
        )


    def get_spillback_resource(self, a, b):
        return self.spillback_resources.get((a, b))



# ============================================================
# 11. JUNCTION MANAGER
# ============================================================

class JunctionManager:

    def __init__(self, env):
        self.env = env
        self.resources = {
            node: simpy.Resource(env, capacity=1)
            for node in JUNCTIONS
        }
        self.activities = []

    def process(self, truck_id, node, from_node, to_node, downstream_spillback_resource=None):
        cfg = JUNCTIONS[node]
        resource = self.resources[node]
        requested_at = self.env.now
        spillback_wait = 0.0
        spillback_request = None

        with resource.request() as request:
            yield request
            service_start = self.env.now
            wait_min = service_start - requested_at

            # For an exit approach, reserve physical queue storage before the
            # truck clears the junction. If the downstream approach is full,
            # the truck remains on the junction resource, reproducing queue
            # spillback and node blocking.
            if downstream_spillback_resource is not None:
                spillback_requested_at = self.env.now
                spillback_request = downstream_spillback_resource.request()
                yield spillback_request
                spillback_wait = self.env.now - spillback_requested_at

            service_min = 60.0 / float(cfg["capacity_vph"])
            yield self.env.timeout(service_min)

        maneuver_delay_min = float(cfg["maneuver_delay_sec"]) / 60.0
        if maneuver_delay_min > 0:
            yield self.env.timeout(maneuver_delay_min)

        finished_at = self.env.now
        self.activities.append({
            "truck_id": truck_id, "junction": node, "from_node": from_node,
            "to_node": to_node, "control": cfg["control"],
            "capacity_vph": cfg["capacity_vph"], "maneuver_delay_sec": cfg["maneuver_delay_sec"],
            "requested_min": requested_at, "start_min": service_start,
            "finish_min": finished_at, "wait_min": wait_min,
            "spillback_wait_min": spillback_wait,
        })
        return spillback_request



# ============================================================
# 12. GATE MANAGER
# ============================================================

class GateManager:
    def __init__(self, env):
        self.env = env
        self.resources = {}
        for gate, cfg in GATE_LANES.items():
            for operation in ("ENTRY", "EXIT"):
                lanes = int(cfg["entry" if operation == "ENTRY" else "exit"])
                if lanes > 0:
                    self.resources[(gate, operation)] = simpy.Resource(env, capacity=lanes)
        self.activities = []
        self.queue_snapshots = []

    def snapshot_queues(self):
        for (gate, operation), resource in self.resources.items():
            self.queue_snapshots.append({
                "time_min": self.env.now,
                "hour": int(self.env.now // 60),
                "gate": gate,
                "operation": operation,
                "queue": len(resource.queue),
                "busy": resource.count,
                "capacity_lanes": resource.capacity,
            })

    def monitor_queues(self):
        while self.env.now <= 24 * 60:
            self.snapshot_queues()
            yield self.env.timeout(QUEUE_SNAPSHOT_INTERVAL_MIN)

    def process(self, truck_id, gate, operation, terminal, cargo):
        service_sec = float(
            GATE_TIMES_SEC[
                ("FULL_ENTRY" if cargo == "FULL" else "EMPTY_ENTRY")
                if operation == "ENTRY"
                else ("FULL_EXIT" if cargo == "FULL" else "EMPTY_EXIT")
            ]
        )
        key = (gate, operation)
        if key not in self.resources:
            raise RuntimeError(f"Gate {gate} has no lanes configured for {operation}.")
        requested_at = self.env.now
        resource = self.resources[key]
        with resource.request() as request:
            yield request
            service_start = self.env.now
            wait_min = service_start - requested_at
            yield self.env.timeout(service_sec / 60.0)
            finished_at = self.env.now
        self.activities.append({
            "truck_id": truck_id, "gate": gate, "operation": operation,
            "cargo": cargo, "service_sec": service_sec,
            "requested_min": requested_at, "start_min": service_start,
            "finish_min": finished_at, "wait_min": wait_min,
        })


# ============================================================
# 12. PATHFINDING
# ============================================================

def find_path(
    graph,
    start,
    destination
):

    try:

        return nx.shortest_path(

            graph,

            source=start,

            target=destination,

            weight="travel_time_min"
        )

    except nx.NetworkXNoPath:

        raise RuntimeError(

            f"No route found from "
            f"{start} to {destination}"
        )


# ============================================================
# 13. TRUCK SIMULATION
# ============================================================

def simulate_truck(
    env,
    truck,
    graph,
    road_manager,
    gate_manager,
    junction_manager,
    route_rng,
    route_overrides=None,
):

    # --------------------------------------------------------
    # Wait until scheduled arrival
    # --------------------------------------------------------

    delay = (
        truck["arrival_min"]
        - env.now
    )


    if delay > 0:

        yield env.timeout(
            delay
        )


    truck["sim_start_min"] = (
        env.now
    )


    terminal = (
        truck["terminal"]
    )


    cargo = (
        truck["cargo"]
    )


    entry_gate = (
        truck["entry_gate"]
    )


    exit_gate = (
        truck["exit_gate"]
    )


    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    destination = (
        get_destination_node(

            terminal,

            truck[
                "mpt_destination"
            ]
        )
    )


    truck[
        "network_destination"
    ] = destination


    # --------------------------------------------------------
    # ENTRY GATE
    # --------------------------------------------------------

    entry_gate_start = (
        env.now
    )


    yield env.process(

        gate_manager.process(

            truck[
                "truck_id"
            ],

            entry_gate,

            "ENTRY",

            terminal,

            cargo
        )
    )


    entry_gate_finish = (
        env.now
    )


    truck[
        "entry_gate_wait_min"
    ] = (
        entry_gate_finish
        - entry_gate_start
        - (
            GATE_TIMES_SEC["FULL_ENTRY" if cargo == "FULL" else "EMPTY_ENTRY"]
        )
        / 60
    )


    # --------------------------------------------------------
    # ENTRY ROAD
    # --------------------------------------------------------

    entry_path = choose_network_path(
        graph,
        entry_gate,
        destination,
        route_rng,
        route_overrides=route_overrides,
    )


    truck[
        "entry_path"
    ] = " > ".join(
        entry_path
    )


    entry_road_start = (
        env.now
    )


    for i in range(
        len(entry_path) - 1
    ):

        if i > 0 and entry_path[i] in JUNCTIONS:
            yield env.process(
                junction_manager.process(
                    truck["truck_id"],
                    entry_path[i],
                    entry_path[i - 1],
                    entry_path[i + 1],
                )
            )

        yield env.process(

            road_manager.traverse(

                truck[
                    "truck_id"
                ],

                entry_path[i],

                entry_path[
                    i + 1
                ]
            )
        )


    entry_road_finish = (
        env.now
    )


    # --------------------------------------------------------
    # TERMINAL PROCESS
    #
    # No terminal queue is modelled.
    # --------------------------------------------------------

    terminal_start = (
        env.now
    )


    process_time = (
        TERMINAL_PROCESS_MIN[
            terminal
        ][
            cargo
        ]
    )


    yield env.timeout(
        process_time
    )


    internal_crossing = (
        INTERNAL_GATE_CROSSING_SEC[
            terminal
        ]
    )


    if internal_crossing > 0:

        yield env.timeout(
            internal_crossing / 60
        )


    terminal_finish = (
        env.now
    )


    # --------------------------------------------------------
    # EXIT ROAD
    # --------------------------------------------------------

    exit_path = choose_network_path(
        graph,
        destination,
        exit_gate,
        route_rng,
        route_overrides=route_overrides,
    )


    truck[
        "exit_path"
    ] = " > ".join(
        exit_path
    )


    exit_road_start = (
        env.now
    )


    for i in range(
        len(exit_path) - 1
    ):

        if i > 0 and exit_path[i] in JUNCTIONS:
            downstream_resource = None
            if i == len(exit_path) - 2:
                downstream_resource = road_manager.get_spillback_resource(exit_path[i], exit_path[i + 1])
            spillback_request = yield env.process(
                junction_manager.process(
                    truck["truck_id"], exit_path[i], exit_path[i - 1], exit_path[i + 1],
                    downstream_spillback_resource=downstream_resource,
                )
            )
            if spillback_request is not None:
                truck["_exit_spillback_request"] = spillback_request

        yield env.process(
            road_manager.traverse(

                truck[
                    "truck_id"
                ],

                exit_path[i],

                exit_path[
                    i + 1
                ]
            )
        )


    exit_road_finish = (
        env.now
    )


    # --------------------------------------------------------
    # EXIT GATE
    # --------------------------------------------------------

    exit_gate_start = env.now

    # If the final gate approach does not have an explicit junction, still
    # reserve its physical storage so the gate queue occupies the road link.
    if "_exit_spillback_request" not in truck and len(exit_path) >= 2:
        a_sp, b_sp = exit_path[-2], exit_path[-1]
        spill_res = road_manager.get_spillback_resource(a_sp, b_sp)
        if spill_res is not None:
            req_sp = spill_res.request()
            spill_requested_at = env.now
            yield req_sp
            truck["_exit_spillback_request"] = req_sp
            truck["exit_spillback_wait_min"] = env.now - spill_requested_at

    yield env.process(
        gate_manager.process(

            truck[
                "truck_id"
            ],

            exit_gate,

            "EXIT",

            terminal,

            cargo
        )
    )


    exit_gate_finish = env.now

    # Release the physical queue-storage slot only after the truck has
    # completed exit-gate processing and left the approach link.
    if truck.get("_exit_spillback_request") is not None and len(exit_path) >= 2:
        road_manager.get_spillback_resource(exit_path[-2], exit_path[-1]).release(truck["_exit_spillback_request"])
        truck.pop("_exit_spillback_request", None)

    truck["exit_gate_wait_min"] = (
        exit_gate_finish
        - exit_gate_start
        - (
            GATE_TIMES_SEC["FULL_EXIT" if cargo == "FULL" else "EMPTY_EXIT"]
        )
        / 60
    )


    truck["exit_spillback_wait_min"] = float(truck.get("exit_spillback_wait_min", 0.0))
    # Junction-held spillback waiting is captured in the junction activity log.

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    truck[
        "sim_finish_min"
    ] = (
        exit_gate_finish
    )


    truck[
        "total_time_min"
    ] = (
        exit_gate_finish
        - truck[
            "sim_start_min"
        ]
    )


    truck[
        "entry_road_time_min"
    ] = (
        entry_road_finish
        - entry_road_start
    )


    truck[
        "terminal_time_min"
    ] = (
        terminal_finish
        - terminal_start
    )


    truck[
        "exit_road_time_min"
    ] = (
        exit_road_finish
        - exit_road_start
    )


# ============================================================
# 14. GATE QUEUE OUTPUT
# ============================================================

def build_gate_queue_dataframe(gate_manager):
    return pd.DataFrame(gate_manager.queue_snapshots)


def build_gate_queue_stock_dataframe(gate_manager):
    """
    Reconstruct gate queue stock from completed gate activities.

    queue_waiting = trucks waiting for a gate resource at the snapshot time.
    in_service = trucks currently occupying a gate lane.
    queue_stock = waiting + in_service.

    The stock view is useful for identifying sustained capacity pressure: if
    requested gate activities accumulate faster than completed activities, the
    stock remains elevated until service capacity catches up.
    """
    activities = pd.DataFrame(gate_manager.activities)
    snapshots = pd.DataFrame(gate_manager.queue_snapshots)

    if snapshots.empty:
        return snapshots
    if activities.empty:
        out = snapshots.copy()
        out["queue_waiting"] = out["queue"]
        out["in_service"] = out["busy"]
        out["queue_stock"] = out["queue"] + out["busy"]
        return out

    rows = []
    for _, snap in snapshots.iterrows():
        gate = snap["gate"]
        operation = snap["operation"]
        t = float(snap["time_min"])
        g = activities[(activities["gate"] == gate) & (activities["operation"] == operation)]
        requested = int((g["requested_min"] <= t).sum())
        finished = int((g["finish_min"] <= t).sum())
        in_system = max(0, requested - finished)
        rows.append({
            **snap.to_dict(),
            "completed": finished,
            "requested": requested,
            "queue_waiting": int(snap["queue"]),
            "in_service": int(snap["busy"]),
            "queue_stock": in_system,
        })
    return pd.DataFrame(rows)


def build_junction_dataframe(junction_manager):
    return pd.DataFrame(junction_manager.activities)


def build_road_hourly_dataframe(road_manager):
    df = pd.DataFrame(road_manager.traversals)
    if df.empty:
        return df

    df["hour"] = (df["requested_min"] // 60).astype(int)
    grouped = (
        df.groupby(
            ["road_id", "physical_road", "from_node", "to_node", "hour"],
            as_index=False,
        )
        .agg(
            flow_vph=("truck_id", "size"),
            capacity_vph=("capacity_vph", "first"),
            lanes=("lanes", "first"),
            distance_km=("distance_km", "first"),
            avg_wait_min=("wait_min", "mean"),
        )
    )
    grouped["vc"] = grouped["flow_vph"] / grouped["capacity_vph"].replace(0, np.nan)
    grouped["vc"] = grouped["vc"].fillna(0.0)
    grouped["los"] = grouped["vc"].apply(los_from_vc)
    return grouped


# ============================================================
# 14. ROAD OUTPUT
# ============================================================

# Screening LOS bands based on the V/C thresholds in the user-provided
# two-lane highway table (level terrain, 0% no-passing zone). The table is
# used here only as a V/C classification framework; it is NOT used to derive
# the 30 km/h road capacity. Capacity remains the explicit model assumption.
LOS_THRESHOLDS = {
    "A": 0.15,
    "B": 0.27,
    "C": 0.43,
    "D": 0.64,
    "E": 1.00,
}


def los_from_vc(vc):
    vc = float(vc)
    for level, threshold in LOS_THRESHOLDS.items():
        if vc <= threshold:
            return level
    return "F"


def build_road_dataframe(
    road_manager
):

    df = pd.DataFrame(
        road_manager.traversals
    )


    if df.empty:

        return df


    rows = []


    grouped = df.groupby(

        [
            "road_id",

            "from_node",

            "to_node"
        ]
    )


    for (
        road_id,
        from_node,
        to_node
    ), group in grouped:

        capacity = (
            group[
                "capacity_vph"
            ].iloc[0]
        )


        hourly = (

            group.assign(

                hour=(
                    group[
                        "requested_min"
                    ]
                    // 60
                ).astype(int)
            )

            .groupby("hour")
            .size()
        )


        peak_flow = (

            int(
                hourly.max()
            )

            if len(hourly) > 0

            else 0
        )


        vc = (

            peak_flow
            / capacity

            if capacity > 0

            else 0
        )


        rows.append(

            {

                "road_id":
                    road_id,

                "from_node":
                    from_node,

                "to_node":
                    to_node,

                "distance_km":
                    group[
                        "distance_km"
                    ].iloc[0],

                "lanes":
                    group[
                        "lanes"
                    ].iloc[0],

                "capacity_vph":
                    capacity,

                "total_traversals":
                    len(group),

                "peak_hourly_flow":
                    peak_flow,

                "peak_vc":
                    vc,

                "los":
                    los_from_vc(vc),

                "avg_wait_min":
                    group[
                        "wait_min"
                    ].mean(),

                "max_wait_min":
                    group[
                        "wait_min"
                    ].max(),

                "avg_travel_min":
                    group[
                        "travel_min"
                    ].mean()
            }
        )


    return pd.DataFrame(
        rows
    )


def build_spillback_dataframe(graph, junctions_df, trucks_df):
    """Summarise exit-gate spillback pressure by gate approach."""
    rows = []
    approach = {"G8": "N06", "G9": "N02", "G1": "N01"}
    for gate, upstream in approach.items():
        if not graph.has_edge(upstream, gate):
            # KML may use N1/N01 naming depending on the network version.
            candidates = [n for n in graph.nodes if str(n).replace("0", "") == upstream.replace("0", "")]
            upstream_node = candidates[0] if candidates else upstream
        else:
            upstream_node = upstream
        if graph.has_edge(upstream_node, gate):
            data = graph[upstream_node][gate]
            storage = int(data.get("capacity_vph", 0) * data.get("travel_time_min", 0) / 60.0)
        else:
            storage = 0
        j = junctions_df[junctions_df["to_node"] == gate] if not junctions_df.empty and "to_node" in junctions_df else pd.DataFrame()
        spill_wait = float(j["spillback_wait_min"].max()) if not j.empty and "spillback_wait_min" in j else 0.0
        rows.append({"Gate": gate, "Upstream node": upstream_node, "Approach storage (trucks)": storage, "Peak spillback wait (min)": spill_wait})
    return pd.DataFrame(rows)


# ============================================================
# 15. KML OUTPUT
# ============================================================

def vc_color(
    vc
):
    # Google Earth uses ABGR. LOS colours follow the requested A-F bands.
    level = los_from_vc(vc)
    colors = {
        "A": "ff5caf3a",  # green
        "B": "ff7fc65a",  # green-orange
        "C": "ff3aa6f0",  # orange
        "D": "ff2f73e0",  # orange-red
        "E": "ff2733d6",  # red
        "F": "ff202080",  # dark congestion
    }
    return colors[level]


def write_congestion_kml(
    graph,
    roads_df
):

    lookup = {}


    for _, row in roads_df.iterrows():

        lookup[
            (
                row[
                    "from_node"
                ],

                row[
                    "to_node"
                ]
            )
        ] = row[
            "peak_vc"
        ]


    lines = []


    lines.append(
        '<?xml version="1.0" encoding="UTF-8"?>'
    )

    lines.append(
        '<kml xmlns="http://www.opengis.net/kml/2.2">'
    )

    lines.append(
        "<Document>"
    )

    lines.append(
        "<name>JIP Traffic Simulation V0.4.5</name>"
    )


    for a, b, data in graph.edges(
        data=True
    ):

        vc = lookup.get(
            (
                a,
                b
            ),
            0.0
        )


        color = vc_color(
            vc
        )


        lines.append(
            "<Placemark>"
        )


        lines.append(
            "<name>"
            f"{data['road_id']} | "
            f"{a} -> {b} | "
            f"V/C {vc:.2f} | LOS {los_from_vc(vc)}"
            "</name>"
        )


        lines.append(
            "<Style>"
        )

        lines.append(
            "<LineStyle>"
        )

        lines.append(
            f"<color>{color}</color>"
        )

        lines.append(
            "<width>6</width>"
        )

        lines.append(
            "</LineStyle>"
        )

        lines.append(
            "</Style>"
        )


        lines.append(
            "<LineString>"
        )

        lines.append(
            "<tessellate>1</tessellate>"
        )

        lines.append(
            "<coordinates>"
        )


        for lon, lat in data[
            "geometry"
        ]:

            lines.append(
                f"{lon},{lat},0"
            )


        lines.append(
            "</coordinates>"
        )

        lines.append(
            "</LineString>"
        )

        lines.append(
            "</Placemark>"
        )


    lines.append(
        "</Document>"
    )

    lines.append(
        "</kml>"
    )


    KML_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def write_hourly_congestion_kml(graph, road_hourly_df):
    """Export a Google Earth KML with one folder per simulation hour."""
    lookup = {}
    if not road_hourly_df.empty:
        for _, row in road_hourly_df.iterrows():
            lookup[(int(row["hour"]), row["from_node"], row["to_node"])] = float(row["vc"])

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '<name>JIP Hourly Congestion V0.4.5</name>',
    ]

    for hour in range(24):
        lines.append(f'<Folder><name>{hour:02d}:00 - {hour:02d}:59</name>')
        seen = set()
        for a, b, data in graph.edges(data=True):
            physical = data["physical_road"]
            if physical in seen:
                continue
            seen.add(physical)
            # Show the worse direction on the shared physical road.
            vc_ab = lookup.get((hour, a, b), 0.0)
            vc_ba = lookup.get((hour, b, a), 0.0)
            vc = max(vc_ab, vc_ba)
            color = vc_color(vc)
            coords = data["geometry"]
            lines.extend([
                '<Placemark>',
                f'<name>{physical} | {a} ↔ {b} | V/C {vc:.2f}</name>',
                '<Style><LineStyle>',
                f'<color>{color}</color><width>7</width>',
                '</LineStyle></Style>',
                '<LineString><tessellate>1</tessellate><coordinates>',
                ' '.join(f'{lon},{lat},0' for lon, lat in coords),
                '</coordinates></LineString>',
                '</Placemark>',
            ])
        lines.append('</Folder>')

    lines.append('</Document></kml>')
    HOURLY_KML_OUTPUT.write_text('\n'.join(lines), encoding='utf-8')


# ============================================================
# 16. VALIDATION
# ============================================================

def validate_configuration():

    if not 0 <= ROUTE_RANDOMIZATION <= 1:
        raise ValueError("ROUTE_RANDOMIZATION must be between 0 and 1.")

    if not 0 <= ARRIVAL_TIME_RANDOMIZATION <= 1:
        raise ValueError(
            "ARRIVAL_TIME_RANDOMIZATION must be between 0 and 1."
        )

    for (terminal, cargo), options in ENTRY_ACCESS_RULES.items():
        if terminal not in CARGO_SHARES:
            raise ValueError(f"Unknown terminal in ENTRY_ACCESS_RULES: {terminal}")
        if cargo not in CARGOS:
            raise ValueError(f"Unknown cargo in ENTRY_ACCESS_RULES: {cargo}")
        total = sum(float(share) for _, share in options)
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"ENTRY_ACCESS_RULES {terminal} {cargo} must sum to 1.0; got {total:.6f}")
        for gate, share in options:
            if gate not in GATE_LANES:
                raise ValueError(f"ENTRY_ACCESS_RULES {terminal} {cargo} references unknown gate {gate}.")
            if share < 0:
                raise ValueError(f"ENTRY_ACCESS_RULES {terminal} {cargo} has negative share for {gate}.")
            if share > 0 and int(GATE_LANES.get(gate, {}).get("entry", 0)) <= 0:
                raise ValueError(f"ENTRY_ACCESS_RULES {terminal} {cargo} assigns flow to {gate}, but {gate} has 0 entry lanes configured.")

    for rule_name, rules in [
        ("ENTRY_GATE_RULES", ENTRY_GATE_RULES),
        ("EXIT_GATE_RULES", EXIT_GATE_RULES),
    ]:
        for flow, options in rules.items():
            if not options:
                raise ValueError(f"{rule_name} {flow} has no options.")
            total = sum(float(share) for _, share in options)
            if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"{rule_name} {flow} must sum to 1.0; got {total:.6f}"
                )
            for gate, share in options:
                if gate not in GATE_LANES:
                    raise ValueError(
                        f"{rule_name} {flow} references unknown gate {gate}."
                    )
                if share < 0:
                    raise ValueError(
                        f"{rule_name} {flow} has negative share for {gate}."
                    )

    for gate, cfg in GATE_LANES.items():
        if int(cfg.get("entry", 0)) < 0 or int(cfg.get("exit", 0)) < 0:
            raise ValueError(f"Gate {gate} lane counts must be >= 0")

    for gate, storage in EXIT_GATE_QUEUE_STORAGE_TRUCKS.items():
        if gate not in GATE_LANES:
            raise ValueError(f"Unknown exit gate in EXIT_GATE_QUEUE_STORAGE_TRUCKS: {gate}")
        if int(storage) <= 0:
            raise ValueError(f"Exit gate queue storage for {gate} must be > 0")

    terminal_total = sum(
        TERMINAL_SHARES.values()
    )


    if not math.isclose(
        terminal_total,
        1.0
    ):

        raise ValueError(
            "TERMINAL_SHARES must sum to 1.0"
        )


    for node, cfg in JUNCTIONS.items():
        if not isinstance(node, str) or not node:
            raise ValueError("Junction node names must be non-empty strings.")
        if float(cfg["capacity_vph"]) <= 0:
            raise ValueError(f"Junction {node} capacity must be > 0")
        if float(cfg["maneuver_delay_sec"]) < 0:
            raise ValueError(f"Junction {node} maneuver delay must be >= 0")

    for terminal, shares in (
        CARGO_SHARES.items()
    ):

        total = sum(
            shares.values()
        )


        if not math.isclose(
            total,
            1.0
        ):

            raise ValueError(

                f"CARGO_SHARES for "
                f"{terminal} must sum to 1.0"
            )


# ============================================================
# 17. SUMMARY
# ============================================================

def print_summary(
    graph,
    trucks_df,
    gates_df,
    roads_df
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "JEDDAH ISLAMIC PORT "
        "TRAFFIC SIMULATION V1.1"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    print(
        f"\nTruck movements: "
        f"{TRUCK_MOVEMENTS_PER_DAY:,}"
    )


    print(
        f"Internal movements: "
        f"{INTERNAL_MOVEMENTS_PER_DAY:,}"
    )


    print(
        f"Total movements including internal: "
        f"{TRUCK_MOVEMENTS_PER_DAY + INTERNAL_MOVEMENTS_PER_DAY:,}"
    )


    print(
        f"\nEntry gate activities: "
        f"{len(gates_df[gates_df['operation'] == 'ENTRY']):,}"
    )


    print(
        f"Exit gate activities: "
        f"{len(gates_df[gates_df['operation'] == 'EXIT']):,}"
    )


    print(
        f"Total gate activities: "
        f"{len(gates_df):,}"
    )


    # --------------------------------------------------------
    # Gates
    # --------------------------------------------------------

    print(
        "\n--- GATE FLOWS ---"
    )


    for gate in GATE_LANES:

        gate_data = gates_df[
            gates_df[
                "gate"
            ] == gate
        ]


        entry = gate_data[
            gate_data[
                "operation"
            ] == "ENTRY"
        ]


        exit_data = gate_data[
            gate_data[
                "operation"
            ] == "EXIT"
        ]


        avg_entry_wait = (

            entry[
                "wait_min"
            ].mean()

            if len(entry) > 0

            else 0
        )


        max_entry_wait = (

            entry[
                "wait_min"
            ].max()

            if len(entry) > 0

            else 0
        )


        avg_exit_wait = (

            exit_data[
                "wait_min"
            ].mean()

            if len(exit_data) > 0

            else 0
        )


        max_exit_wait = (

            exit_data[
                "wait_min"
            ].max()

            if len(exit_data) > 0

            else 0
        )


        print(

            f"{gate}: "

            f"Entry {len(entry):,}, "

            f"Exit {len(exit_data):,}, "

            f"Total {len(gate_data):,}, "

            f"Avg entry wait "
            f"{avg_entry_wait:.2f} min, "

            f"Max entry wait "
            f"{max_entry_wait:.2f} min, "

            f"Avg exit wait "
            f"{avg_exit_wait:.2f} min, "

            f"Max exit wait "
            f"{max_exit_wait:.2f} min"
        )


    # --------------------------------------------------------
    # Terminal flows
    # --------------------------------------------------------

    print(
        "\n--- TERMINAL FLOWS ---"
    )


    terminal_counts = (
        trucks_df[
            "terminal"
        ]
        .value_counts()
        .sort_index()
    )


    for terminal, count in (
        terminal_counts.items()
    ):

        print(
            f"{terminal}: "
            f"{count:,}"
        )


    # --------------------------------------------------------
    # Truck times
    # --------------------------------------------------------

    print(
        "\n--- TRUCK TIMES ---"
    )


    print(
        f"Average truck time: "
        f"{trucks_df['total_time_min'].mean():.2f} min"
    )


    print(
        f"Median truck time: "
        f"{trucks_df['total_time_min'].median():.2f} min"
    )


    print(
        f"Maximum truck time: "
        f"{trucks_df['total_time_min'].max():.2f} min"
    )


    # --------------------------------------------------------
    # Road network
    # --------------------------------------------------------

    print(
        "\n--- ROAD NETWORK ---"
    )


    print(
        f"Physical roads: "
        f"{len(graph.edges()) // 2}"
    )


    if not roads_df.empty:

        print(
            "\nHighest V/C links:"
        )


        highest = (
            roads_df
            .sort_values(
                "peak_vc",
                ascending=False
            )
            .head(15)
        )


        for _, row in (
            highest.iterrows()
        ):

            print(

                f"{row['from_node']} -> "
                f"{row['to_node']} | "

                f"Peak flow "
                f"{int(row['peak_hourly_flow']):,} vph | "

                f"Capacity "
                f"{int(row['capacity_vph']):,} vph | "

                f"V/C "
                f"{row['peak_vc']:.2f} | "

                f"Avg wait "
                f"{row['avg_wait_min']:.2f} min"
            )


    # --------------------------------------------------------
    # Hourly arrivals
    # --------------------------------------------------------

    print(
        "\n--- HOURLY ARRIVALS ---"
    )


    trucks_df[
        "arrival_hour"
    ] = (

        trucks_df[
            "arrival_min"
        ]
        // 60
    ).astype(int)


    hourly = (
        trucks_df
        .groupby(
            "arrival_hour"
        )
        .size()
    )


    for hour in range(24):

        count = int(
            hourly.get(
                hour,
                0
            )
        )


        print(
            f"{hour:02d}:00 - "
            f"{hour:02d}:59 : "
            f"{count:,}"
        )


    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    print(
        "\n--- CHECKS ---"
    )


    entries = len(
        gates_df[
            gates_df[
                "operation"
            ] == "ENTRY"
        ]
    )


    exits = len(
        gates_df[
            gates_df[
                "operation"
            ] == "EXIT"
        ]
    )


    if entries == TRUCK_MOVEMENTS_PER_DAY:

        print(
            "OK - 10,500 entry activities"
        )

    else:

        print(
            "ERROR - entry activity count"
        )


    if exits == TRUCK_MOVEMENTS_PER_DAY:

        print(
            "OK - 10,500 exit activities"
        )

    else:

        print(
            "ERROR - exit activity count"
        )


    if (
        entries + exits
        == TRUCK_MOVEMENTS_PER_DAY * 2
    ):

        print(
            "OK - 21,000 total gate activities"
        )

    else:

        print(
            "ERROR - total gate activity count"
        )


# ============================================================
# 18. MAIN
# ============================================================

def run_simulation(
    trucks_per_day=None,
    route_randomization=None,
    seed=None,
    entry_gate_rules=None,
    exit_gate_rules=None,
    entry_access_rules=None,
    gate_lanes=None,
    average_speed_kmh=None,
    lanes_per_direction=None,
    capacity_per_lane_vph=None,
    arrival_time_variability=None,
    junctions=None,
    appointment_management_enabled=None,
    max_entries_per_hour=None,
    route_overrides=None,
    logistics_destination_daily=None,
    gate_times_sec=None,
    n2_n1_corridor_lanes=None,
    terminal_shares=None,
    cargo_shares=None,
    save_outputs=True,
):
    global TRUCK_MOVEMENTS_PER_DAY, INTERNAL_MOVEMENTS_PER_DAY, ROUTE_RANDOMIZATION, RANDOM_SEED
    global ENTRY_GATE_RULES, EXIT_GATE_RULES, ENTRY_ACCESS_RULES
    global GATE_LANES, AVERAGE_SPEED_KMH, GATE_TIMES_SEC, N2_N1_CORRIDOR_LANES
    global LANES_PER_DIRECTION, CAPACITY_PER_LANE_VPH, ROAD_CAPACITY_VPH, JUNCTIONS
    global APPOINTMENT_MANAGEMENT_ENABLED, MAX_PORT_ENTRIES_PER_HOUR, ROUTE_OVERRIDES
    global TERMINAL_SHARES, CARGO_SHARES
    global LOGISTICS_DESTINATION_DAILY

    original = (
        TRUCK_MOVEMENTS_PER_DAY,
        INTERNAL_MOVEMENTS_PER_DAY,
        ROUTE_RANDOMIZATION,
        RANDOM_SEED,
        ENTRY_GATE_RULES,
        EXIT_GATE_RULES,
        ENTRY_ACCESS_RULES,
        GATE_LANES,
        AVERAGE_SPEED_KMH,
        LANES_PER_DIRECTION,
        CAPACITY_PER_LANE_VPH,
        ROAD_CAPACITY_VPH,
        JUNCTIONS,
        APPOINTMENT_MANAGEMENT_ENABLED,
        MAX_PORT_ENTRIES_PER_HOUR,
        ROUTE_OVERRIDES,
        LOGISTICS_DESTINATION_DAILY.copy(),
        dict(GATE_TIMES_SEC),
        N2_N1_CORRIDOR_LANES,
        dict(TERMINAL_SHARES),
        {k:dict(v) for k,v in CARGO_SHARES.items()},
    )

    if trucks_per_day is not None:
        TRUCK_MOVEMENTS_PER_DAY = int(trucks_per_day)
    INTERNAL_MOVEMENTS_PER_DAY = int(round(TRUCK_MOVEMENTS_PER_DAY * INTERNAL_MOVEMENTS_SHARE))
    if route_randomization is not None:
        ROUTE_RANDOMIZATION = float(route_randomization)
    if seed is not None:
        RANDOM_SEED = int(seed)
    if arrival_time_variability is None:
        arrival_time_variability = ARRIVAL_TIME_RANDOMIZATION
    else:
        arrival_time_variability = float(arrival_time_variability)
    if not 0 <= arrival_time_variability <= 1:
        raise ValueError(
            "arrival_time_variability must be between 0 and 1."
        )
    if entry_gate_rules is not None:
        ENTRY_GATE_RULES = entry_gate_rules
    if exit_gate_rules is not None:
        EXIT_GATE_RULES = exit_gate_rules
    if entry_access_rules is not None:
        ENTRY_ACCESS_RULES = {k:list(v) for k,v in entry_access_rules.items()}
    if gate_lanes is not None:
        GATE_LANES = dict(gate_lanes)
    if average_speed_kmh is not None:
        AVERAGE_SPEED_KMH = float(average_speed_kmh)
    if lanes_per_direction is not None:
        LANES_PER_DIRECTION = int(lanes_per_direction)
    if capacity_per_lane_vph is not None:
        CAPACITY_PER_LANE_VPH = float(capacity_per_lane_vph)
    ROAD_CAPACITY_VPH = LANES_PER_DIRECTION * CAPACITY_PER_LANE_VPH
    if junctions is not None:
        JUNCTIONS = {k: dict(v) for k, v in junctions.items()}
    if appointment_management_enabled is not None:
        APPOINTMENT_MANAGEMENT_ENABLED = bool(appointment_management_enabled)
    if max_entries_per_hour is not None:
        MAX_PORT_ENTRIES_PER_HOUR = int(max_entries_per_hour)
    if logistics_destination_daily is not None:
        LOGISTICS_DESTINATION_DAILY={str(k):int(v) for k,v in logistics_destination_daily.items()}
    if route_overrides is not None:
        ROUTE_OVERRIDES = {tuple(k): list(v) for k, v in route_overrides.items()}
    if gate_times_sec is not None:
        GATE_TIMES_SEC = {k: float(v) for k, v in gate_times_sec.items()}
    if n2_n1_corridor_lanes is not None:
        N2_N1_CORRIDOR_LANES = int(n2_n1_corridor_lanes)
    if terminal_shares is not None:
        TERMINAL_SHARES = {str(k): float(v) for k, v in terminal_shares.items()}
    if cargo_shares is not None:
        CARGO_SHARES = {str(k): {str(c): float(v) for c, v in shares.items()} for k, shares in cargo_shares.items()}

    try:
        validate_configuration()

        print("\nReading KML network...")
        routes = read_kml_routes()
        print(f"Routes read: {len(routes)}")
        for route in routes:
            print(
                f"  {route['name']} | "
                f"{route['distance_km']:.3f} km | "
                f"{route['travel_time_min']:.2f} min"
            )

        graph = build_network(routes)
        route_errors = validate_route_overrides(graph, ROUTE_OVERRIDES)
        if route_errors:
            raise ValueError("\nInvalid route overrides:\n" + "\n".join(f"  - {e}" for e in route_errors))
        print(f"\nNetwork nodes: {graph.number_of_nodes()}")
        print(f"Directed links: {graph.number_of_edges()}")

        required_nodes = {
            "G1", "G2", "G4", "G6", "G8", "G9",
            "N01", "N02", "N04", "N05", "N06", "N07", "N08", "N09", "N10",
            "DPW", "RSGT", "MPT1", "MPT2", "MPT3",
            "Logipoint", "CMA CGM", "Bahri Logistics",
        }
        missing_nodes = required_nodes - set(graph.nodes)
        if missing_nodes:
            raise RuntimeError(
                "\nThe following required nodes are missing from the KML:\n"
                + str(sorted(missing_nodes))
            )

        # ------------------------------------------------------------
        # NETWORK CONNECTIVITY CHECK
        # ------------------------------------------------------------
        routing_pairs = set()

        for (terminal, cargo), options in ENTRY_ACCESS_RULES.items():
            destinations = (
                list(MPT_DESTINATION_SPLIT.keys()) if terminal == "MPT"
                else list(LOGISTICS_DESTINATION_DAILY.keys()) if terminal == "LOGISTICS"
                else [get_destination_node(terminal)]
            )
            for destination_raw in destinations:
                destination = get_destination_node(
                    terminal,
                    destination_raw if terminal == "MPT" else None,
                )
                for gate, share in options:
                    if float(share) > 0:
                        routing_pairs.add((gate, destination))

        for (terminal, cargo), options in EXIT_GATE_RULES.items():
            destinations = (
                list(MPT_DESTINATION_SPLIT.keys()) if terminal == "MPT"
                else list(LOGISTICS_DESTINATION_DAILY.keys()) if terminal == "LOGISTICS"
                else [get_destination_node(terminal)]
            )
            for destination_raw in destinations:
                destination = get_destination_node(
                    terminal,
                    destination_raw if terminal == "MPT" else None,
                )
                for gate, share in options:
                    if float(share) > 0:
                        routing_pairs.add((destination, gate))

        connectivity_errors = []
        for start, destination in sorted(routing_pairs):
            if start not in graph or destination not in graph:
                connectivity_errors.append(
                    f"{start} -> {destination}: missing node"
                )
            elif not nx.has_path(graph, start, destination):
                connectivity_errors.append(
                    f"{start} -> {destination}: no route"
                )

        if connectivity_errors:
            raise RuntimeError(
                "\nNETWORK CONNECTIVITY CHECK FAILED:\n"
                + "\n".join(f"  - {x}" for x in connectivity_errors)
            )

        print("\nNETWORK CONNECTIVITY CHECK: OK")

        print("\nGenerating truck demand...")
        trucks, appointment_df = create_trucks(
            arrival_time_variability=arrival_time_variability,
            appointment_management_enabled=APPOINTMENT_MANAGEMENT_ENABLED,
            max_entries_per_hour=MAX_PORT_ENTRIES_PER_HOUR,
            logistics_destination_daily=LOGISTICS_DESTINATION_DAILY,
        )
        print(f"Trucks generated: {len(trucks):,}")

        env = simpy.Environment()
        road_manager = RoadManager(env, graph)
        gate_manager = GateManager(env)
        junction_manager = JunctionManager(env)
        route_rng = random.Random(RANDOM_SEED + 100003)

        env.process(gate_manager.monitor_queues())

        for truck in trucks:
            env.process(
                simulate_truck(
                    env,
                    truck,
                    graph,
                    road_manager,
                    gate_manager,
                    junction_manager,
                    route_rng,
                    ROUTE_OVERRIDES,
                )
            )

        print("\nRunning simulation...")
        env.run()
        print("Simulation finished.")

        trucks_df = pd.DataFrame(trucks)
        gates_df = pd.DataFrame(gate_manager.activities)
        roads_df = build_road_dataframe(road_manager)
        road_hourly_df = build_road_hourly_dataframe(road_manager)
        queues_df = build_gate_queue_dataframe(gate_manager)
        queue_stock_df = build_gate_queue_stock_dataframe(gate_manager)
        junctions_df = build_junction_dataframe(junction_manager)

        spillback_df = build_spillback_dataframe(graph, junctions_df, trucks_df)

        if save_outputs:
            trucks_df.to_csv(TRUCK_OUTPUT, index=False)
            gates_df.to_csv(GATE_OUTPUT, index=False)
            roads_df.to_csv(ROAD_OUTPUT, index=False)
            queues_df.to_csv(
                BASE_DIR / "simulation_gate_queues_v060.csv",
                index=False,
            )
            queue_stock_df.to_csv(
                BASE_DIR / "simulation_gate_queue_stock_v060.csv",
                index=False,
            )
            junctions_df.to_csv(JUNCTION_OUTPUT, index=False)
            write_congestion_kml(graph, roads_df)
            write_hourly_congestion_kml(graph, road_hourly_df)

        print_summary(graph, trucks_df, gates_df, roads_df)

        return {
            "trucks": trucks_df,
            "gates": gates_df,
            "roads": roads_df,
            "road_hourly": road_hourly_df,
            "queues": queues_df,
            "queue_stock": queue_stock_df,
            "junctions": junctions_df,
            "spillback": spillback_df,
            "appointment_profile": appointment_df,
            "graph": graph,
        }
    finally:
        (
            TRUCK_MOVEMENTS_PER_DAY,
            INTERNAL_MOVEMENTS_PER_DAY,
            ROUTE_RANDOMIZATION,
            RANDOM_SEED,
            ENTRY_GATE_RULES,
            EXIT_GATE_RULES,
            ENTRY_ACCESS_RULES,
            GATE_LANES,
            AVERAGE_SPEED_KMH,
            LANES_PER_DIRECTION,
            CAPACITY_PER_LANE_VPH,
            ROAD_CAPACITY_VPH,
            JUNCTIONS,
            APPOINTMENT_MANAGEMENT_ENABLED,
            MAX_PORT_ENTRIES_PER_HOUR,
            ROUTE_OVERRIDES,
            LOGISTICS_DESTINATION_DAILY,
            GATE_TIMES_SEC,
            N2_N1_CORRIDOR_LANES,
            TERMINAL_SHARES,
            CARGO_SHARES,
        ) = original


def main():
    result = run_simulation()

    print("\nOutput files:")
    print(f"  {TRUCK_OUTPUT.name}")
    print(f"  {GATE_OUTPUT.name}")
    print(f"  {ROAD_OUTPUT.name}")
    print(f"  {KML_OUTPUT.name}")
    print(f"  {JUNCTION_OUTPUT.name}")
    print(f"  {HOURLY_KML_OUTPUT.name}")
    print("  simulation_gate_queues_v060.csv")
    print("  simulation_gate_queue_stock_v060.csv")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()