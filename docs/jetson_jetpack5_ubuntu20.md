# Jetson JetPack 5 — Ubuntu 20.04 example

This guide shows how to run **slam_icp** on an NVIDIA Jetson Nano (2 GB) with
**JetPack 5.x** (Ubuntu 20.04). The package is **self-contained**: wheel
odometry, scan frame fix, minimal URDF, and SLAM live inside `slam_icp` — no
`odom_to_tf`, `puzzlebot_description`, or Gazebo required on the robot.

| Goal | Recommended approach |
|------|----------------------|
| SLAM on a **real Puzzlebot / LiDAR robot** | ROS 2 **Humble** + `slam_icp_jetson_launch.py` |
| Native Humble on Jetson | **JetPack 6** (Ubuntu 22.04) — run `install_deps_jetson_ubuntu20.sh` |
| JetPack 5 (Ubuntu 20.04) | Humble in **Docker** (`ros:humble-ros-base`) — see Example B |
| Full **Gazebo simulation** on Jetson | Not recommended (heavy); use a desktop PC |

---

## Example A — On-robot SLAM (Puzzlebot + RPLidar, no Gazebo)

Typical stack on JetPack 5 + Ubuntu 20.04:

```
RPLidar driver   →  /scan
wheel encoders   →  /VelocityEncL, /VelocityEncR
slam_icp launch  →  /odom, TF, /map, /slam_pose
```

Bundled nodes inside `slam_icp`:

| Node | Role |
|------|------|
| `wheel_odometry` | encoders → `/odom` + TF `odom → base_footprint` |
| `scan_republisher` | `/scan` → `/scan_fixed` with `frame_id=laser` |
| `robot_state_publisher` | minimal URDF `base_footprint → laser` |
| `slam_icp_node` | SLAM + TF `map → odom` |

### 1. Install dependencies

```bash
cd ~/ros2_ws/src/slam_icp
chmod +x scripts/install_deps_jetson_ubuntu20.sh
./scripts/install_deps_jetson_ubuntu20.sh
```

Or follow the [Humble install guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
(Ubuntu 22.04 / JetPack 6).

Initialize rosdep once:

```bash
sudo rosdep init   # skip if already done
rosdep update
```

### 2. Copy slam_icp to the Jetson and build

Only this package is needed in `~/ros2_ws/src/`:

```bash
mkdir -p ~/ros2_ws/src
# scp/rsync/git clone slam_icp into ~/ros2_ws/src/
cd ~/ros2_ws

source /opt/ros/humble/setup.bash
rosdep install -i --from-path src --rosdistro humble -y
colcon build --packages-select slam_icp
source install/setup.bash
```

> **JetPack 5 (20.04):** there are no native `ros-humble-*` debs. Use Docker
> (Example B) or upgrade to JetPack 6.

### 3. Python scientific stack

Prefer system packages on Jetson:

```bash
sudo apt install -y python3-numpy python3-scipy python3-opencv python3-yaml
```

If OpenCV is missing `distanceTransform`, use pip (Jetson wheels may vary):

```bash
pip3 install --user -r ~/ros2_ws/src/slam_icp/requirements.txt
```

### 4. Launch (after RPLidar + encoder drivers are running)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch slam_icp slam_icp_jetson_launch.py
```

Optional RViz (heavy on 2 GB — prefer a remote PC with the same `ROS_DOMAIN_ID`):

```bash
ros2 launch slam_icp slam_icp_jetson_launch.py rviz:=true
```

Tune wheel geometry if needed:

```bash
ros2 launch slam_icp slam_icp_jetson_launch.py wheel_radius:=0.05 wheel_base:=0.19
```

Parameters tuned for Nano 2 GB live in `config/slam_icp_jetson.yaml` (fewer
particles, lower publish rate, smaller map).

### 5. TF chain checklist

```
map  →  odom            slam_icp_node
odom →  base_footprint   wheel_odometry
base_footprint → laser    robot_state_publisher (urdf/puzzlebot_minimal.urdf)
```

Do **not** run a second node that publishes `odom → base_footprint`.

### 6. Autostart on boot (systemd example)

Create `/etc/systemd/system/slam-icp.service`:

```ini
[Unit]
Description=slam_icp Jetson stack
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jetson
Environment="ROS_DOMAIN_ID=0"
ExecStart=/bin/bash -lc 'source /opt/ros/humble/setup.bash && source /home/jetson/ros2_ws/install/setup.bash && ros2 launch slam_icp slam_icp_jetson_launch.py'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable after adjusting paths and ensuring LiDAR/encoder drivers start first:

```bash
sudo systemctl daemon-reload
sudo systemctl enable slam-icp.service
sudo systemctl start slam-icp.service
```

---

## Example B — Humble in Docker on Jetson (Ubuntu 20.04 host)

Use this when you want the exact same Humble environment as a desktop PC.

```bash
docker pull ros:humble-ros-base

docker run -it --rm --net=host \
  -v ~/ros2_ws:/ws \
  ros:humble-ros-base bash

# Inside the container
apt update && apt install -y python3-pip python3-colcon-common-extensions \
  ros-humble-robot-state-publisher
pip3 install -r /ws/src/slam_icp/requirements.txt
cd /ws && colcon build --packages-select slam_icp
source install/setup.bash
ros2 launch slam_icp slam_icp_jetson_launch.py
```

Pass device nodes (`--device=/dev/ttyUSB0`) if the LiDAR is USB-attached.

---

## Example C — JetPack 6 / Ubuntu 22.04 (native Humble)

JetPack 6 uses Ubuntu 22.04; use `scripts/install_deps_ubuntu22.sh` and the
same `slam_icp_jetson_launch.py` launch file.

---

## Performance tips on Jetson Nano 2 GB

- Defaults in `slam_icp_jetson.yaml` already reduce particles, beams, and map size.
- Set `publish_rate_hz` to 2–3 if CPU is saturated.
- Keep `rviz:=false` on the Nano; visualize from a laptop.
- Enable `pose_graph_runtime.optimize_in_thread: true` (default) to avoid blocking scans during loop closure.

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| No `/map` | Check `/scan` and encoder topics; verify `use_sim_time:=false` |
| TF errors in RViz | Run `ros2 run tf2_tools view_frames`; ensure only one `odom→base_footprint` publisher |
| Empty scan in SLAM | Confirm RPLidar driver is running; check `ros2 topic hz /scan` |
| `cv2` import error | Install JetPack OpenCV or `python3-opencv` from apt |
| High CPU / lag | Lower particles and `publish_rate_hz` in `slam_icp_jetson.yaml` |
