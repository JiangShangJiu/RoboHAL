# RoboArena

机器人学习工程：数据采集与训练结果可视化。基于 Franka Panda MuJoCo 仿真。

## 结构

```
RoboArena/
├── assets/franka_emika_panda/   # Panda MuJoCo 模型
├── src/
│   ├── simulation/              # 仿真环境
│   ├── planning/                # 运动规划（OMPL 关节空间，可扩展其他后端）
│   ├── data_collection/         # 数据采集
│   └── visualization/           # 可视化
├── data/                        # 采集数据（gitignore）
├── outputs/                     # 训练输出（gitignore）
└── scripts/                     # 入口脚本
```

## 安装

```bash
pip install -r requirements.txt
```

### 可选：OMPL 关节空间规划（无 ROS）

默认用 **OMPL** 在关节空间做采样规划（不依赖 ROS / MoveIt2，体量更小）。安装可选依赖：

```bash
pip install -e ".[planning]"
# 或: pip install 'ompl>=1.6'
```

规划结果可导出为与数据采集相同的 HDF5，再用 `replay_trajectory.py` 在 MuJoCo 中回放。

```bash
python scripts/plan_joint_ompl.py --goal 0,0,0,-1.5,0,1.5,0.7 -o data/ompl_plan.h5
python scripts/replay_trajectory.py data/ompl_plan.h5 --episode 0
```

代码组织：`src/planning/protocols.py` 只定义统合的 `Planner` 协议（`plan_joint` / `plan_cartesian`，不依赖 OMPL）；`OMPLJointPlanner` 为关节空间 OMPL 实现，`PandaOMPLJointPlanner` 为其 Panda 默认限位封装；若以后接入 **MoveIt2** 或笛卡尔插值+IK，可再增实现类（仍通过 `Planner` 对接数据采集/仿真）。

**说明：** 部分环境下 OMPL 的 Python 绑定在进程退出时可能触发底层释放告警；若遇异常，可在独立子进程/脚本中调用规划，或关注 OMPL 与 Python 版本组合。

## 使用

### 场景切换

支持多个场景描述文件，通过 `--scene` 或环境变量 `ROBOARENA_SCENE` 指定：

```bash
# 列出可用场景
python scripts/show_env.py --list-scenes

# 指定场景
python scripts/show_env.py --scene mjx_single_cube.xml
```

自定义场景可放在 `assets/scenes/` 下，会自动被发现。

### 展示仿真环境

查看数据采集时使用的 Panda 仿真场景，空格切换静止/随机探索：

```bash
python scripts/show_env.py
python scripts/show_env.py --scene mjx_single_cube.xml
```

### 数据采集

```bash
python scripts/collect_data.py --episodes 10 --steps 500 --output data/demo.h5
```

### 轨迹回放

```bash
python scripts/replay_trajectory.py data/demo.h5 --episode 0
```

### 训练曲线可视化

将 TensorBoard 日志或 `metrics.csv` 放在 `outputs/` 下，然后：

```bash
python scripts/visualize_training.py outputs/run1 --output plot.png
```

## 代码示例

录制与回放统一由 `TrajectoryManager` 管理：

```python
from src.data_collection import TrajectoryManager
from src.data_collection.policies import RandomPolicy

mgr = TrajectoryManager()
policy = RandomPolicy(nu=mgr.env.nu)

# 录制
mgr.record(policy, num_episodes=5, max_steps_per_episode=500)
mgr.save("data/demo.h5")

# 回放
mgr.load("data/demo.h5")
mgr.replay(episode_idx=0, slowdown=2.0)
```

## 环境变量

- `ROBOARENA_PANDA_PATH`: 自定义 Panda 模型路径（默认使用 `assets/franka_emika_panda`）
- `ROBOARENA_SCENE`: 默认场景名（如 `scene.xml`、`mjx_single_cube.xml`）
