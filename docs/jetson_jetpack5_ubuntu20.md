# Jetson JetPack 5 — Ubuntu 20.04 example

This guide shows how to run **slam_icp** on an NVIDIA Jetson with **JetPack 5.x**
(Ubuntu 20.04). JetPack 5 does not ship ROS 2 Humble natively (Humble targets
Ubuntu 22.04), so pick the path that matches your hardware.

| Goal | Recommended approach |
|------|----------------------|
| SLAM on a **real Puzzlebot / LiDAR robot** | ROS 2 Foxy + pip wheels on Jetson |
| Full **Gazebo simulation** on Jetson | Not recommended (heavy); use a desktop PC |
| Same code as desktop Humble | Docker with `ros:humble` on Jetson, or upgrade to JetPack 6 (22.04) |

---

## Example A — On-robot SLAM (Puzzlebot + RPLidar, no Gazebo)

Typical stack on JetPack 5 + Ubuntu 20.04:

```
RPLidar driver  →  /scan
wheel odometry  →  /odom
robot_state_publisher  →  TF odom → base_footprint → laser_frame
slam_icp_node   →  /map, /slam_pose, TF map → odom
```

### 1. Install ROS 2 Foxy (Ubuntu 20.04 native)

Follow the [official Foxy install guide](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html), then:

```bash
sudo apt update
sudo apt install -y \
  ros-foxy-rclpy \
  ros-foxy-sensor-msgs \
  ros-foxy-nav-msgs \
  ros-foxy-geometry-msgs \
  ros-foxy-tf2-ros \
  ros-foxy-tf2-geometry-msgs \
  python3-colcon-common-extensions \
  python3-rosdep
```

Initialize rosdep once:

```bash
sudo rosdep init   # skip if already done
rosdep update
```

### 2. Clone and build slam_icp

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/richy-gs/slam_icp.git
cd ~/ros2_ws

source /opt/ros/foxy/setup.bash
rosdep install -i --from-path src --rosdistro foxy -y
colcon build --packages-select slam_icp
source install/setup.bash
```

> **Foxy vs Humble:** This package is developed on Humble. Foxy uses the same
> `rclpy` APIs for the topics and parameters used here; if you hit a Foxy-specific
> error, check ROS 2 release notes or prefer the Docker/Humble path below.

### 3. Install Python scientific stack on Jetson

Prefer system packages when available:

```bash
sudo apt install -y python3-numpy python3-scipy python3-opencv python3-yaml
```

If apt versions are too old or OpenCV is missing `distanceTransform`, use pip
(Jetson often needs pre-built wheels):

```bash
pip3 install --user -r ~/ros2_ws/src/slam_icp/requirements.txt
```

On some Jetson images you may need NVIDIA/community wheels for OpenCV; if
`import cv2` fails after pip, install the Jetson OpenCV package from JetPack
and keep only `numpy`/`scipy` from pip.

### 4. Tune frames for your robot

Edit `config/slam_icp.yaml` (or override at launch):

```yaml
slam_icp_node:
  ros__parameters:
    use_sim_time: false
    tf:
      base_frame: "base_footprint"
      odom_frame: "odom"
      map_frame: "map"
      scan_frame: "laser_frame"    # Puzzlebot LiDAR frame
```

### 5. Launch SLAM only

After your robot drivers publish `/scan`, `/odom`, and TF:

```bash
source /opt/ros/foxy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run slam_icp slam_icp_node --ros-args \
  --params-file ~/ros2_ws/install/slam_icp/share/slam_icp/config/slam_icp.yaml
```

Optional RViz on the Jetson (or from a remote machine with `ROS_DOMAIN_ID` set):

```bash
ros2 run rviz2 rviz2 -d ~/ros2_ws/install/slam_icp/share/slam_icp/config/rviz_slam.rviz
```

### 6. Autostart on boot (systemd example)

Create `/etc/systemd/system/slam-icp.service`:

```ini
[Unit]
Description=slam_icp node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jetson
Environment="ROS_DOMAIN_ID=0"
ExecStart=/bin/bash -lc 'source /opt/ros/foxy/setup.bash && source /home/jetson/ros2_ws/install/setup.bash && ros2 run slam_icp slam_icp_node --ros-args --params-file /home/jetson/ros2_ws/install/slam_icp/share/slam_icp/config/slam_icp.yaml'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable after adjusting paths and ensuring robot drivers start first:

```bash
sudo systemctl daemon-reload
sudo systemctl enable slam-icp.service
sudo systemctl start slam-icp.service
```

---

## Example B — Humble in Docker on Jetson (Ubuntu 20.04 host)

Use this when you want the exact same Humble environment as a desktop PC.

```bash
# On the Jetson host
docker pull ros:humble-ros-base

docker run -it --rm --net=host \
  -v ~/ros2_ws:/ws \
  ros:humble-ros-base bash

# Inside the container
apt update && apt install -y python3-pip python3-colcon-common-extensions
pip3 install -r /ws/src/slam_icp/requirements.txt
cd /ws && colcon build --packages-select slam_icp
source install/setup.bash
ros2 run slam_icp slam_icp_node --ros-args \
  --params-file install/slam_icp/share/slam_icp/config/slam_icp.yaml
```

Pass device nodes (`--device=/dev/ttyUSB0`) if the LiDAR is USB-attached.

---

## Example C — JetPack 6 / Ubuntu 22.04 (native Humble)

JetPack 6 uses Ubuntu 22.04; follow the main [README](../README.md) without
changes. This is the simplest path if you can upgrade the board image.

---

## Performance tips on Jetson

- Lower `mcl.num_particles` and `scan.max_beams` in `slam_icp.yaml`.
- Set `publish_rate_hz` to 2–5 on Nano-class boards.
- Enable `pose_graph_runtime.optimize_in_thread: true` (default) to avoid
  blocking the scan callback during loop-closure optimization.
- Run Gazebo simulation on a workstation, not on the Jetson, unless you only
  need headless testing.

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| Node stuck in `WAITING_TF` | Check `robot_state_publisher` and frame names in YAML |
| Empty `/map` | Verify `/scan` ranges and `use_sim_time` matches your clock |
| `cv2` import error | Install JetPack OpenCV or `python3-opencv` from apt |
| High CPU / lag | Reduce particles, beams, and `publish_rate_hz` |
