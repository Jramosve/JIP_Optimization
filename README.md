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
The fast gate Monte Carlo (`monte_carlo.py`) reproduces the main.py demand generation (terminal split, hourly profile and variability, ELM appointment cap, cargo mix, entry/exit gate allocation, MPT destinations), gate lanes and processing times, and the full truck cycle that feeds the exit gates (entry gate → route → terminal process → route → exit gate). All scenarios of a batch are simulated together with numpy, so 1,000 scenarios of 10,500 trucks run in a few seconds.

It reports P10/P50/P90/P95/P99 of peak gate queue (trucks waiting + in service, sampled every 5 min as in main.py), peak gate queue time and average gate queue time, per gate and operation, next to the value from the current simulation run. Road-link and junction queueing are not simulated, so results only diverge from main.py when the internal network itself is saturated.

## Run locally
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
