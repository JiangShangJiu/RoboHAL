# RoboArena

机器人学习工程：数据采集与训练结果可视化。基于 Franka Panda MuJoCo 仿真。

## 结构

```
RoboArena/
├── assets/franka_emika_panda/   # Panda MuJoCo 模型
├── src/
│   ├── simulation/              # 仿真环境
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
