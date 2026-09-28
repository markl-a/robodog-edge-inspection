# Deploying on a real dog: the plan, not done here

None of this was run on hardware. It is the path I would take, and the questions I would ask first.

## Target

- **Robot:** a quadruped with a payload bay. Examples are the Unitree Go2/B2 or a DEEP Robotics X30; the choice depends on IP rating, stairs and battery life.
- **Compute:** NVIDIA Jetson Orin NX / Nano, mounted in the payload bay.
- **Sensors:** an RGB camera on a pan-tilt head, plus a radiometric thermal core. A Lepton-class core is fine for breakers at 0.5–1 m; farther targets need higher resolution.

## Model path

1. Keep ONNX as the interchange format. The INT8 model here uses **QDQ explicit quantization**, which TensorRT consumes directly: `trtexec --onnx=gauge_int8.onnx --int8 --saveEngine=gauge.plan`.
2. Build the engine **on the target device**, because TensorRT engines are not portable across GPUs or TensorRT versions. Pin JetPack in the image.
3. Re-measure latency and accuracy on the device. The CPU numbers in the README are a laptop's and say nothing about Orin.
4. Set the power mode (`nvpmodel`) deliberately. Inference competes with locomotion and navigation for the same battery.

## Robot integration

- `edge/mission.py` depends only on the `Robot` protocol: `go_to`, `capture`, `battery_pct`, `dock`. On the real robot I would implement it as a ROS 2 node. Navigation goals would go through Nav2 or the vendor SDK. Captures would be triggered once the robot reports it has stopped and settled, because motion blur while settling is a real error source.
- Commissioning captures a **reference image per waypoint** plus gauge and ROI calibration. Registration and ROIs depend on it, so it has to be re-done whenever equipment is moved.

## Operations

- **Outbox:** the SQLite file lives on persistent storage (not tmpfs), so results survive a reboot mid-patrol.
- **Clock:** `capturedAt` comes from the robot, which is time-synced over NTP when docked. The platform orders readings by capture time, not arrival time.
- **Model updates:** version the models, and record the model version in each result's `detail` so the platform can compare accuracy across versions.

## Questions for the team

- Which is the bigger problem today: misreads, missed checkpoints (navigation or battery), or data that never arrives?
- Who reviews `NEEDS_REVIEW` images, and how fast? That sets the confidence threshold.
- Are alarm limits owned by the customer's facility team or by us?
