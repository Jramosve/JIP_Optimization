# Jeddah Islamic Port – Truck Traffic Model V1.1

V1.1 updates the JIP truck traffic screening model with the latest validated operating inputs and exit-gate queue spillback into the internal road network.

## Base inputs
- Gate truck movements/day: 10,500 base; 15,000 mid-term; 20,000 long-term; custom up to 30,000.
- Internal movements: 5% of gate truck traffic; not routed through Mawani gates.
- Terminal split: RSGT 40%, DP World 35%, MPT 20%, Logistics 5%.
- Full/Empty: 70% / 30% for all terminals.
- Terminal processing: RSGT 75 min, DP World 60 min, MPT 75 min, Logistics 75 min.
- Gate processing: Full IN 35 s, Full OUT 30 s, Empty IN 20 s, Empty OUT 20 s.
- Gate lanes: G1 2+1, G4 3+0, G6 0+0, G8 0+3, G9 3+2.
- Road capacity: 800 veh/h/lane; average speed 30 km/h.
- Junction capacity: 700 veh/h; junction-specific values remain editable.
- N4–N2 corridor: 3 lanes/direction base, switchable to 4.
- ELM appointment release cap: 500 entries/hour, editable.
- Appointment compliance: 100%; released trucks are distributed approximately uniformly within each hour.
- Hourly demand profile: observed 24-hour profile supplied for JIP.

## V1.1 network enhancement
Exit-gate queues now occupy physical storage on the final approach links to G8, G9 and G1. When the approach storage is full, upstream junctions can be held by the queued exit traffic, creating spillback into the internal network.

The dashboard separately reports gate queue time and exit-gate spillback pressure.

## Monte Carlo
The fast gate Monte Carlo (`monte_carlo.py`) reproduces the main.py demand generation (terminal split, hourly profile and variability, ELM appointment cap, cargo mix, entry/exit gate allocation, MPT destinations, route choice), gate lanes and processing times, and the full truck cycle that feeds the exit gates (entry gate → route → terminal process → route → exit gate). All scenarios of a batch are simulated together with numpy, so 1,000 scenarios of 10,500 trucks run in a few seconds.

It reports P10/P50/P90/P95/P99 of peak gate queue (trucks waiting + in service, sampled every 5 min as in main.py), peak and average gate queue time per gate and operation, and the external marshalling-area backlog, next to the value from the current simulation run. Road-link and junction queueing are not simulated, so results only diverge from main.py when the internal network itself is saturated. Road lanes, capacity per lane and the N4–N2 corridor lanes only affect the full simulation.

## Al-Khumra logistics corridor
Trucks travelling directly between the Al-Khumra logistics corridor and the port are modelled as a **subset of the daily demand**, never as additional trucks. For each terminal (RSGT, DPW, MPT, Logistics) the user defines, separately, how many trucks **enter** the port from Al-Khumra and how many **leave** towards it, either as a share of that terminal's daily trucks or as trucks/day, plus the entry gate and exit gate used by the corridor (`AL_KHUMRA_FLOWS` in `main.py`: `entry_value`, `exit_value`, `entry_gate`, `exit_gate`; editable in the dashboard). Example: with 20,000 trucks/day and 4,000 entering from Al-Khumra, 4,000 entries use the corridor entry gate and the remaining 16,000 follow the general gate allocation.

The entry and exit selections are **nested**: trucks are ranked once per terminal, so the trucks entering from Al-Khumra are the first ones to leave towards it (with 20% in and 30% out, the 20% make the round trip and another 10% only leave towards Al-Khumra). With equal entry and exit values every Al-Khumra truck makes the round trip. Older configurations with a single `value` apply it to both directions.

Optionally, each terminal can set the **share of its Al-Khumra trucks that are empty** (`empty_share`, 0–1; blank or `None` = any cargo, as before). The Al-Khumra trucks are then picked among the terminal's existing empty and full trucks in that proportion; no truck changes cargo, so the terminal's overall Full/Empty split is unchanged and the rest of its traffic becomes correspondingly fuller. An error is raised if the terminal does not have enough empty (or full) trucks. In the model each truck has a single Full/Empty state for its whole visit (entry and exit).

Tagged trucks keep their terminal, cargo, destination and arrival time; only the gate of the tagged direction changes. They share gate lanes and queues with other traffic and remain subject to the ELM appointment cap. The selection uses a separate random stream, so with all values at 0 the model is identical to the run without Al-Khumra tagging, and the non-Al-Khumra trucks are identical across Al-Khumra scenarios with the same seed. Truck outputs include `origin` (`AL_KHUMRA` if the truck uses the corridor in either direction, otherwise `GENERAL`) and the flags `al_khumra_entry` / `al_khumra_exit`; the Monte Carlo applies the same rule and reports `al_khumra_entry_trucks`, `al_khumra_exit_trucks`, `al_khumra_trucks` and `al_khumra_empty_trucks` per scenario.

## Appointment management and marshalling area
With the ELM cap enabled, trucks that cannot enter in their scheduled hour wait in the external marshalling area and are released FIFO in later hours. The dashboard shows hourly demand vs releases, the marshalling-area queue over the day, per-truck waiting time (`scheduled_arrival_min`, `marshalling_wait_min`) and when the backlog clears. The **ELM cap comparison** runs the Monte Carlo for several caps to show the trade-off between gate queues and marshalling-area waiting.

## Network map
The congestion map can be shown on satellite imagery (Esri), a street map (CARTO) or as the offline schematic diagram. Satellite and street tiles need internet access in the browser.

## Tests
```bash
pip install pytest
pytest -q
```
The tests check that the Monte Carlo reproduces main.py (same truck list → identical entry-gate queues), that main.py runs fall inside the Monte Carlo range, and that every dashboard input changes the Monte Carlo output.

## Run locally
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
