# robodog-edge-inspection

**The edge-AI part of a quadruped patrol robot for plant rooms: it reads analog pressure gauges, checks breakers with a thermal camera, and gets results home without losing any. It is built around the failure modes a walking robot actually has.**

This is a portfolio demo from September 2026. All images are **synthetic**, rendered with the nuisances a robot camera sees: glare, focus blur, sensor noise, a cable in front of the dial, and pose error at each stop. The robot is a simulator behind a small interface. **Nothing here has run on a real robot or a Jetson**; [docs/deploy.md](docs/deploy.md) describes that path. Results are sent to [patrol-platform-spring](https://github.com/markl-a/patrol-platform-spring), which owns alarm limits and work orders.

```
 robot at waypoint ──► capture ──► gauge reader / thermal check ──► result (measurement + confidence)
                          ▲                 │ low confidence                     │
                          └──── re-capture ◄┘                                    ▼
                                                              SQLite outbox (store-and-forward, idempotency key)
                                                                                 │ link up
                                                                                 ▼
                                                     patrol-platform-spring: limits → alarms / reviews → work orders
```

## Results

### 1. Gauge reading (0–10 bar, EN 837-1 class 1.6 → tolerance ±0.16 bar)

| Reader | Normal images: within class | Hard images*: within class | Hard: gross errors (>1 bar) | Latency (1 thread) | Size |
|---|---|---|---|---|---|
| Classic CV (Hough + darkest ray) | 96.2 % | 92.7 % | 13 / 1000 | 0.41 ms | — |
| **Classic + confidence gate** (≥0.3, else human review) | 96.2 % (0 % sent to review) | 93.9 % (5.8 % sent to review) | **0** | 0.41 ms | — |
| CNN FP32 (ONNX) | **99.8 %** | 50.9 % | 332 / 1000 | 0.43 ms | 760 KB |
| CNN INT8 (ONNX Runtime static QDQ, per-channel) | 99.6 % | 50.4 % | 334 / 1000 | **0.21 ms** | **205 KB** |

\* The hard images have heavy glare and, in half of them, a cable or pipe crossing the dial. **The CNN was trained without occluders**, so this row measures behaviour outside its training distribution.

Numbers are from `results/gauge_report.json`. The CNN has 194 k params and trained in 12 epochs, taking 209 s on a laptop CPU. ONNX output matches PyTorch within 2e-6.

**What this shows**
- **The best model on clean data was the worst in the field.** The CNN is more accurate on images like its training set. Outside that set it is confidently wrong one time in three, and it has no way to say "I'm not sure".
- **The classic reader can say it is unsure.** Its confidence measures how much the needle stands out from the next-darkest direction. A cable across the dial creates a second dark ray and drives confidence toward 0. In the mission, a low-confidence reading is **re-captured from a slightly different pose** (parallax moves the cable, not the needle). If it is still unclear, it is sent as `NEEDS_REVIEW`, never guessed.
- **Known remaining failure** ([results/failure_case.png](results/failure_case.png)): a cable almost parallel to the needle, with glare washing out the needle, read 3.2 bar instead of 9.3 at confidence 0.49. It was 2 of 6,000 hard images. The platform catches this class of error a second way: a reading that jumps more than `max_jump` from the last patrol opens a *verify* task next to any alarm.
- **INT8 is a clear win on the edge**: 3.7× smaller and 2× faster, for 0.2 points of accuracy. What I'd do next for the CNN is add occlusion augmentation plus an uncertainty signal (an ensemble or MC-dropout). Until then it stays behind the same review gate.

### 2. Thermal check of a three-phase breaker (160×120 radiometric, NETA MTS ΔT bands)

| | Severity accuracy | ΔT error | Actionable faults missed (of 144) |
|---|---|---|---|
| Fixed ROIs from commissioning | 61.8 % | 3.07 °C | **91** |
| **Register to the waypoint reference first** (phase correlation) | **98.5 %** | 0.16 °C | 3 |

A legged robot never stops in exactly the same place; here the pose error is up to ±10 px. A loose lug is a hot spot a few pixels across, so ROIs calibrated at commissioning slide off it. Registering each frame to that waypoint's reference image before reading the ROIs is the difference between catching the fault and missing it. ΔT is compared **between phases**, not against a fixed temperature, because that is how NETA bands are defined and because load changes every phase at once. (`results/thermal_report.json`)

### 3. Mission and uplink (`python -m edge.demo`)

The demo route has 6 checkpoints; one of them is behind a blocked aisle. The plant-room Wi-Fi is down during the patrol.

```
B1-PG-01  OK           4.207   (truth 4.2)
B1-PG-02  OK           3.563   (truth 3.6)
B1-SW-01  OK           dT 0.01 normal
B1-SW-02  OK           dT 7.71 probable, phase C      ← loose lug
B1-PG-03  OK           7.755   (truth 7.9)
B1-PG-04  SKIPPED      reason: unreachable            ← reported, not silently dropped
flush in plant room: sent 0, pending 6   →   flush at dock: sent 6, pending 0
```

- **The robot measures; the platform decides.** Limits live in the platform database, so changing one does not need a robot software release.
- **Battery rule:** before each checkpoint, the robot checks it can still get back to the dock with a 1.3× margin and 15 % reserve. If not, it docks and marks the rest `SKIPPED (battery)`. A missed inspection shows up as a coverage gap, not as "no problem found".
- **Exactly-once delivery:** each result's key is `uuid5(mission, checkpoint)`. Only keys the server confirms leave the outbox, and the server dedupes on the key. A test covers outages and lost ACKs. The same batch was also pushed over HTTP to the running Spring service.

## Run

```bash
pip install -r requirements.txt
python -m edge.demo          # patrol + outage + recovery → results/patrol_demo.json, thermal_report.json
python -m edge.train_gauge   # CNN train → ONNX FP32 → INT8 → compare with classic → results/gauge_report.json
pytest -q                    # 21 tests: gauge maths, confidence gate, NETA bands, registration, mission rules, outbox
```

## Layout

```
edge/gauge.py        GaugeSpec (range, sweep, EN 837 class), synthetic renderer, classic reader + confidence
edge/train_gauge.py  CNN, ONNX export, static INT8 quantization, benchmark
edge/thermal.py      synthetic radiometric frames, registration, per-phase hotspot ΔT, NETA bands
edge/mission.py      Robot interface, SimRobot, MissionRunner (battery rule, retries, review, skips)
edge/uplink.py       SQLite outbox, idempotency keys, HTTP sender
edge/demo.py         end-to-end run
docs/deploy.md       Jetson / TensorRT / ROS 2 path (not tested on hardware)
```

MIT License.
