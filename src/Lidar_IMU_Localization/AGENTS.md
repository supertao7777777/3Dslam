# Repository Guidelines

## Project Structure & Module Organization

- `src/`: core C++17 nodes. `src/lio` contains LIO odometry; `src/loc` contains map-based localization logic; top-level `.cpp` files are legacy entry points.
- `include/`: public headers for estimators, IMU integration, utilities, Sophus, ikd-Tree, and shared message types.
- `config/`: YAML configs (`params.yaml`, `mid360.yaml`) defining topics, sensor settings, and localization options.
- `launch/`: ROS launch files (`run.launch`, `run_loc.launch`) and RViz configs in `launch/include/`.
- `map/`, `Log/`, `doc/`, `msg/`: maps and outputs, logs, diagrams, and custom ROS messages (`cloud_info.msg`, `QRcode.msg`).

## Build, Test, and Development Commands

- Build (from catkin workspace root): `catkin_make -DCMAKE_BUILD_TYPE=Release` (package name: `lio_localization`).
- Setup environment: `source devel/setup.bash` before running any nodes.
- Run LIO odometry: `roslaunch lio_localization run.launch` and play a bag with `rosbag play <bag>.bag --clock`.
- Run localization with prior map: `roslaunch lio_localization run_loc.launch` and play the same bag; monitor RViz outputs.

## Coding Style & Naming Conventions

- C++ only, using C++17, 4-space indentation, and brace-on-same-line style as in `src/lio/featureExtract.cpp`.
- Classes/structs use PascalCase (`FeatureExtract`, `Estimator`); enums follow `SensorType`; helpers and variables use existing lowerCamelCase / snake_case patterns.
- Prefer extending utilities under `include/` instead of duplicating logic; keep headers self-contained and ROS-friendly.

## Testing Guidelines

- Current tests are bag-driven: use the launch files above and inspect trajectories, TF, and point clouds in RViz.
- When adding automated tests, use `rostest`/`rosbag` integration tests under a new `test/` directory, named `test_<feature>.launch`.
- For numerical or optimization changes, validate against known datasets and compare trajectories before and after.

## Commit & Pull Request Guidelines

- Use clear, present-tense commit messages, e.g. `lio: refine feature extraction thresholds` or `loc: fix map alignment`.
- In pull requests, describe datasets used, key parameter changes (with paths like `config/params.yaml`), and include RViz screenshots when possible.
- Avoid large mixed refactors; separate pure formatting from behavioural changes to simplify review.

