# Cruzr S2 接入状态

当前工程**没有安装真实 Cruzr S2 模型**，也没有用其他机器人或简化模型替代它。

用户指定的 [fiveages-sim/robot_usds](https://github.com/fiveages-sim/robot_usds) 列出了
`humanoid/Ubtech/Ubtech_CruzrS2`，其中 `humanoid/Ubtech` 是
[fiveages-sim/ubtech-usds](https://github.com/fiveages-sim/ubtech-usds) 子模块。
2026-09-25 检查时，主仓库 API 可读并指向子模块提交
`c8219fde6affb8ed85de45faf422565b0a50bee4`，子仓库 API 匿名请求返回 HTTP 404，
`git ls-remote https://github.com/fiveages-sim/ubtech-usds.git HEAD` 要求身份认证。
这说明当前环境无法读取资源，不能据此断定仓库已删除还是私有。
追溯到 2026-03-05 的首次公开主仓库记录，Ubtech 也已经是子模块。

主仓库的 Apache-2.0 许可证不自动证明子模块资产适用同一许可证。
需要从模型提供者取得模型及其对应许可，不能把公开的 **Walker S2** 当作 **Cruzr S2**。

## 已准备的本地导入入口

取得真实模型后，在项目根目录执行：

```bash
python scripts/import_cruzr_s2.py \
  --source /path/to/cruzr_s2/model.xml \
  --license /path/to/cruzr_s2/LICENSE
```

也支持普通 URDF；遇到 ROS 包路径时显式映射：

```bash
python scripts/import_cruzr_s2.py \
  --source /path/to/cruzr_description/urdf/cruzr_s2.urdf \
  --package cruzr_description=/path/to/cruzr_description \
  --license /path/to/cruzr_description/LICENSE
```

导入器会编译原始模型、展开 MJCF include，复制引用的 mesh/texture/hfield，
保留提供的许可和文件 SHA-256，重新加载打包后的 MJCF 并检查初始动力学是否有限。
只有这些检查通过后才会创建：

```text
assets/robots/cruzr_s2/cruzr_s2.xml
assets/robots/cruzr_s2/assets/...
assets/robots/cruzr_s2/provenance.json
assets/scenes/cruzr_s2/empty.xml
```

已有安装不会被覆盖。导入器不能自动验证厂家身份，也不把能编译等同于真实硬件精度。
空场景保留原模型的几何和自由度，不自动加地面、浮动基座、执行器或虚构传感器。

## 只有 USD 时

当前验证环境使用 MuJoCo 3.4.0，本脚本不转换 USD。新版 MuJoCo 的
[OpenUSD 导入](https://mujoco.readthedocs.io/en/stable/OpenUSD/importing.html)
仍依赖具体构建和资产兼容性，不能承诺任意 Isaac Sim 模型可直接使用。

可行路线是取得厂家的 URDF/MJCF；或在具备相关 USD/Isaac Sim 工具的环境中导出
URDF/MJCF 及引用网格，保留原始关节、质量、惯量和碰撞参数，再运行以上本地导入器。
若取得子模块访问权限，应按主仓库记录的路径和提交取回完整引用资源：

```bash
git clone --depth 1 https://github.com/fiveages-sim/robot_usds.git
cd robot_usds
git submodule update --init --recursive humanoid/Ubtech
```

URDF 的根在 MuJoCo 中通常固定；移动底盘需要在 MJCF 中明确描述自由度和轮地接触。
本工具拒绝含 mimic、floating 或 planar 关节的 URDF，以免静默丢失这些约束。
URDF 不包含完整的厂商驱动与传感器语义，需在 MJCF/硬件配置中单独验证。
控制接口接通后，还需核对关节正方向、限位、最大速度/力矩、零位、传动比和碰撞行为。
