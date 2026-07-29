# RoboArena

轻量 MuJoCo 仿真：多机器人场景、数据采集、关节规划、轨迹回放。

## 结构

```
assets/robots/<name>/    机器人本体
assets/scenes/<name>/    场景 XML
src/simulation/          SimEnv
src/planning/            OMPL 关节规划（可选）
src/data_collection/     录制 → HDF5
src/visualization/       回放 / 曲线
scripts/                 入口
```

## 安装

```bash
pip install -r requirements.txt
pip install -e ".[planning]"   # 可选 OMPL
```

## 快速开始

```bash
python scripts/show_env.py --list-robots
python scripts/show_env.py --list-scenes
python scripts/show_env.py --scene kitchen_lite.xml

python scripts/collect_data.py --scene kitchen_lite.xml --episodes 10 -o data/demo.h5
python scripts/replay_trajectory.py data/demo.h5 --episode 0

python scripts/plan_joint_ompl.py --goal 0,0,0,-1.5,0,1.5,0.7 -o data/ompl_plan.h5
```

```python
from src.simulation import SimEnv
from src.data_collection import TrajectoryManager
from src.data_collection.policies import RandomPolicy

env = SimEnv(robot="panda", scene="kitchen_lite.xml")
mgr = TrajectoryManager(env=env)
mgr.record(RandomPolicy(nu=env.nu), num_episodes=5, max_steps_per_episode=500)
mgr.save("data/demo.h5")
```

## 机器人

| 名 | 说明 |
|----|------|
| `panda` | Franka + 夹爪（默认） |
| `panda_allegro` | Panda + Allegro 手 |
| `pal_tiago_dual` | TIAGo++ 底盘双臂 |
| `robot_soccer_kit` | 小型全向底盘 |
| `allegro` | 单独手掌（拼装用） |

Panda 场景：`empty.xml` / `kitchen_lite.xml` / `mjx_single_cube.xml`。详见 `assets/README.md`。

## 环境变量

- `ROBOARENA_ROBOT`（默认 `panda`）
- `ROBOARENA_SCENE`（默认 `empty.xml`）
