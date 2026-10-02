# Assets

```text
assets/
  robots/<name>/     # 机器人 XML + mesh；第三方保留 LICENSE
  props/             # 可复用家具 / 物体
  scenes/<name>/     # 场景：include 机器人 + props
```

## 约定

- 场景 include 示例：
  - `<include file="../../robots/pal_tiago_dual/tiago_dual_position.xml"/>`
  - `<include file="../../props/living_mobile_set.xml"/>`
- 加载器会把 robots / props / 场景 XML 合并到临时目录
- 默认：`ROBOHAL_ROBOT`、`ROBOHAL_SCENE`

## 新增场景

1. 需要的家具放进 `props/`
2. `scenes/<robot>/xxx.xml` include 机器人与 props
3. `robohal --robot <name> --scene xxx.xml`

## Props

| 文件 | 说明 |
|------|------|
| `living_mobile_set.xml` | 移动臂客厅家具（中央留空） |

## 场景

| 机器人 | 文件 | 内容 |
|--------|------|------|
| `panda` | `empty.xml` / `kitchen_lite.xml` / `mjx_single_cube.xml` | 固定臂 |
| `pal_tiago_dual` | `empty.xml` / `velocity.xml` / `living_lite.xml` | 移动双臂客厅 |

```bash
robohal --robot pal_tiago_dual --scene living_lite.xml
robohal --robot panda --scene kitchen_lite.xml
```

机器人模型多来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)；`panda_allegro` 为本仓库组合。
