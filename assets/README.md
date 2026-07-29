# Assets

```text
assets/
  robots/<name>/     # XML + assets/ mesh；第三方模型保留各自 LICENSE
  scenes/<name>/     # 场景，include 对应机器人
```

## 约定

- 场景用相对路径 include，例如：`<include file="../../robots/panda/panda.xml"/>`
- 加载器会合并 XML 并改写 `meshdir`
- 默认机器人 / 场景：`ROBOARENA_ROBOT`、`ROBOARENA_SCENE`

## 新增

1. `robots/<name>/` 放本体与 mesh  
2. `scenes/<name>/empty.xml` include 该本体  
3. `python scripts/show_env.py --robot <name>`

## Panda 场景

| 文件 | 内容 |
|------|------|
| `empty.xml` | 空地 |
| `kitchen_lite.xml` | 台面 / 水槽 / 柜 / 可抓物 |
| `mjx_single_cube.xml` | 单方块 |

```bash
python scripts/show_env.py --scene kitchen_lite.xml
```

机器人模型多来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)；`panda_allegro` 为本仓库组合。
