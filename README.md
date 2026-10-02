# RoboHAL

轻量 MuJoCo 机器人硬件仿真：多机器人 / 多场景，一套仿真与真机通用的同步硬件接口。

## 结构

```
assets/robots/<name>/    机器人本体
assets/props/            家具 / 物体
assets/scenes/<name>/    场景 XML
robohal/api.py         后端无关接口：RobotHardware / RobotBackend / RealHardware
robohal/hardware.py    MujocoHardware：仿真实现（+ JointConfig 等数据类在 api.py）
robohal/models.py      资产发现与 MJCF 加载
robohal/cli.py         命令行入口
scripts/                 独立工具（资产导入）
tests/                   行为测试
```

## 安装

```bash
pip install -r requirements.txt
pip install -e .        # 提供 robohal 命令
pip install -e ".[dev]" # 可选：pytest
```

## 快速开始

```bash
robohal --list-robots                      # 列出机器人与场景（不需图形后端）
# 移动双臂客厅（推荐）
robohal --robot pal_tiago_dual --scene living_lite.xml
# 固定臂厨房
robohal --robot panda --scene kitchen_lite.xml
# 无窗口跑一段并打印 JSON 状态
robohal --robot panda --scene empty.xml --headless --steps 1000 --position joint1=0.5
```

## 机器人

| 名 | 说明 |
|----|------|
| `panda` | Franka + 夹爪（默认） |
| `panda_allegro` | Panda + Allegro 手 |
| `pal_tiago_dual` | TIAGo++ 底盘双臂 |
| `robot_soccer_kit` | 小型全向底盘 |
| `allegro` | 单独手掌（拼装用） |

推荐客厅：`--robot pal_tiago_dual --scene living_lite.xml`（移动底盘 + 双臂）。  
家具在 `assets/props/`。详见 `assets/README.md`。

## 硬件接口（仿真 / 真机同一套代码）

`robohal.api` 不依赖 MuJoCo，定义了后端无关的协议与数据类：

- `RobotHardware`：控制器应依赖的同步接口（命名关节、SI 单位、命令闩锁、`step`/`read`、锁存急停）。
- `RobotBackend`：真机侧唯一需要实现的小接口（总线读写、使能/断电、关节描述）。
- `RealHardware`：在 `RobotBackend` 之上补齐校验、闩锁与急停语义。

`MujocoHardware` 是 `RobotHardware` 的仿真实现（结构匹配，`tests/test_api.py` 校验），因此同一份控制代码可直接切到真机：

```python
from robohal import RobotHardware, MujocoHardware

def run(hw: RobotHardware):
    with hw:
        hw.set_positions({"joint1": 0.5})
        return hw.step(250).joints["joint1"].position

run(MujocoHardware.from_robot("panda", "empty.xml"))  # 仿真
# run(RealHardware(MyRobotBackend("/dev/ttyUSB0")))  # 真机，只需实现 RobotBackend
```

真机接入：实现 `RobotBackend`（或直接对照 `robohal/api.py`）即可；实时调度、总线协议与标定由后端负责。

### 从 MJCF / URDF 导入新机器人

```bash
python scripts/import_cruzr_s2.py --source model.urdf --license LICENSE.txt
```

校验通过后写入 `assets/robots/<name>/` 并生成 `provenance.json`；本仓库不内置第三方模型。

## 环境变量

- `ROBOHAL_ROBOT`（默认 `panda`）
- `ROBOHAL_SCENE`（默认 `empty.xml`）
- `ROBOHAL_ASSETS`（外部资产目录，默认仓库内 `assets/`）
