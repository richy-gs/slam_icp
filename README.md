# slam_icp

2D SLAM for **ROS 2 Humble** in pure Python: **ICP** scan matching, **Monte Carlo
Localization (MCL)**, occupancy-grid mapping, and pose-graph loop closure. No
`nav2`, `slam_toolbox`, or other navigation meta-packages.

Tested with **Gazebo Classic 11** + TurtleBot3 (`burger`) and optionally with
**GZ Sim** + Puzzlebot (companion workspace).

## Features

- ICP between consecutive LiDAR scans (and scan-to-map when enabled)
- Particle-filter localization with KLD-adaptive resampling
- Likelihood-field sensor model (distance transform on the live map)
- Occupancy grid builder with ray casting
- Pose graph with loop-closure detection and Levenberg–Marquardt optimization

## Quick start (simulation)

```bash
# 1. Create a workspace and clone this repo
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/richy-gs/slam_icp.git
cd ~/ros2_ws

# 2. Install dependencies
source /opt/ros/humble/setup.bash
rosdep update
rosdep install -i --from-path src --rosdistro humble -y

# 3. Build and source
colcon build --packages-select slam_icp
source install/setup.bash

# 4. Run TurtleBot3 in Gazebo Classic
export TURTLEBOT3_MODEL=burger
ros2 launch slam_icp slam_icp_gazebo_launch.py rviz:=true
```

Useful launch arguments: `world`, `x_pose`, `y_pose`, `rviz`, `use_sim_time`.

## Replicate on another computer

### Supported platforms

| Platform | ROS 2 | Notes |
|----------|-------|-------|
| Ubuntu 22.04 (desktop / JetPack 6) | Humble | Primary target; sim + real robot |
| Ubuntu 20.04 (Jetson JetPack 5) | Humble (Docker) | Native debs unavailable; see [docs/jetson_jetpack5_ubuntu20.md](docs/jetson_jetpack5_ubuntu20.md) |

### Prerequisites

- [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- Python 3.10+
- `colcon`, `rosdep`

Simulation extras (installed automatically by `rosdep` when keys resolve):

```bash
sudo apt install ros-humble-gazebo-ros-pkgs \
                 ros-humble-turtlebot3-gazebo \
                 ros-humble-turtlebot3-description \
                 ros-humble-rviz2
```

Or use the helper script:

```bash
./scripts/install_deps_ubuntu22.sh
```

### Python dependencies

On a normal ROS desktop install, scientific packages come from apt via
`rosdep` (`python3-numpy`, `python3-scipy`, `python3-opencv`, `python3-yaml`).

If you prefer **pip** (venv, Docker, or Jetson):

```bash
pip3 install -r requirements.txt
```

ROS client libraries (`rclpy`, message packages, `tf2_ros`) are **not**
pip-installable; install them with your ROS 2 distribution.

Development / unit tests:

```bash
pip3 install -r requirements-dev.txt
```

### Build

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --packages-select slam_icp
source install/setup.bash
```

### Tests

```bash
# From the package directory (no ROS runtime required for algorithm tests)
cd ~/ros2_ws/src/slam_icp
python3 -m pytest test/ -q

# Or via colcon
cd ~/ros2_ws
colcon test --packages-select slam_icp
colcon test-result --verbose
```

## Launch files

| Launch file | Simulator | Robot | Extra deps |
|-------------|-----------|-------|------------|
| `slam_icp_gazebo_launch.py` | Gazebo Classic 11 | TurtleBot3 burger | `turtlebot3_gazebo` (via rosdep) |
| `slam_icp_puzzlebot_launch.py` | GZ Sim (Ignition) | Puzzlebot + LiDAR | External `puzzlebot_gazebo` workspace |
| `slam_icp_jetson_launch.py` | none (real robot) | Puzzlebot + RPLidar | **none** — self-contained |

### Puzzlebot simulation (optional)

The Puzzlebot launch expects a companion workspace (e.g.
`TE3003B_Eq2/ros2_ws`) built and sourced **before** this package:

```bash
# Build companion ws first, then slam_icp in the same overlay
source ~/puzzlebot_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch slam_icp slam_icp_puzzlebot_launch.py rviz:=true
```

## Run on a real robot (Jetson / Puzzlebot)

Copy **only** `slam_icp` to the robot workspace. On JetPack 5 (Ubuntu 20.04)
see [docs/jetson_jetpack5_ubuntu20.md](docs/jetson_jetpack5_ubuntu20.md).

```bash
# After RPLidar + encoder drivers are running:
ros2 launch slam_icp slam_icp_jetson_launch.py
```

This launch starts wheel odometry, scan frame fix, minimal URDF TF, and SLAM.
Parameters for Nano 2 GB: `config/slam_icp_jetson.yaml`.

SLAM-only mode (if you already provide `/odom` and TF yourself):

```bash
ros2 run slam_icp slam_icp_node --ros-args \
  --params-file $(ros2 pkg prefix slam_icp)/share/slam_icp/config/slam_icp_jetson.yaml
```

## Topic and TF contract

| Direction | Topic / TF | Type |
|-----------|------------|------|
| Subscribe | `/scan` | `sensor_msgs/LaserScan` |
| Subscribe | `/odom` | `nav_msgs/Odometry` |
| Publish | `/map` | `nav_msgs/OccupancyGrid` |
| Publish | `/slam_pose` | `geometry_msgs/PoseStamped` |
| Publish | `/particlecloud` | `geometry_msgs/PoseArray` |
| Publish | `/slam_path`, `/odom_path` | `nav_msgs/Path` |
| TF | `map` → `odom` | broadcast by node |

Parameters live in `config/slam_icp.yaml`. RViz layout: `config/rviz_slam.rviz`.

## Package layout

```
slam_icp/
├── slam_icp/          # Python modules (icp, mcl, map_builder, pose_graph, robot/, …)
├── urdf/              # puzzlebot_minimal.urdf (on-robot TF, no external description pkg)
├── launch/            # Gazebo Classic, Puzzlebot sim, Jetson real-robot
├── config/            # slam_icp.yaml, slam_icp_jetson.yaml, rviz_slam.rviz
├── test/              # pytest unit tests
├── docs/              # Platform-specific guides (Jetson)
├── scripts/           # install_deps_ubuntu22.sh, install_deps_jetson_ubuntu20.sh
├── package.xml
├── setup.py
├── requirements.txt
└── requirements-dev.txt
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
