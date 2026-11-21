# SpaceExp session handoff

更新时间：2026-11-15（America/Los_Angeles）。本文件用于让新 session 继续当前项目；请先核对实际文件，保留用户之后的修改。

## 1. 用户偏好与最近的要求

- 使用中文交流；**所有结果表格使用英文列名、模型名和类别名**。
- 代码尽量简洁，不增加无关抽象。脚本可调参数放在文件顶部，用大写变量或配置字典，不使用 argparse。
- 用户经常亲自修改入口配置、移动文件。不要恢复用户删除或移动的文件，不要重置工作区，不要自行提交、推送或执行 `update.sh`。
- 有明确授权的工作直接执行，不反复询问确认。不要擅自运行付费 API、全量 benchmark、重建或训练。
- **不再计算或展示“轨迹中曾到达目标就算成功”的指标（Ever SR）**。用户明确要求忽略它。
- 保留现有 near/medium/far **软配额**：每场景尽量 2/2/2，但可以出现 3/2/1。用户已明确决定不改成硬上限。
- 不要将训练、导航任务逻辑重新放入 `agent/`，也不要重新引入独立的 `navigation.py`。
- **最近用户要求记住四模型的 32 行完整分组表。** 之后说“展示四模型完整分组表”，就展示本文件末尾或下述已保存版本，不要换回只有 Overall 误差的宽表。

## 2. 项目目的

项目目录：`/data4/haibo/code/SpaceExp`。

任务是图像目标导航：模型从一张 initial view 对应的真实相机位姿出发，通过相机工具控制虚拟相机，最终到达 target 的位置和朝向。

场景事先由视频通过 VGGT-Omega + DA3 重建。SpaceExp 运行时只加载已有 `reconstruction.ply` 和 `cameras.json`，由后台常驻 Blender 渲染，不调用重建模型。模型获得第一人称观察图。

每组 initial → target 有两个兄弟任务：

- `first_person`：目标图是原视频的第一人称帧，评估位置和完整 forward 方向。
- `top_down`：目标图是静态俯视地图，绿色 Start 和箭头、橙色 Target 和箭头；评估位置和水平 heading，忽略俯仰与 roll。

未来用户打算利用原视频相机轨迹生成可学习的导航轨迹，再 SFT 小模型，特别是 Qwen3.5-9B。**尚未实施 SFT 轨迹导出或训练。**

## 3. 环境、实际入口和代码职责

Python：`/home/hw139426/workspace/miniconda3/envs/spatial_agent/bin/python`

Blender：`/data4/haibo/software/blender/blender-4.5.13-linux-x64/blender`

普通检查通常使用 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1`；绘图可用 `MPLCONFIGDIR=/tmp/spaceexp-matplotlib`。

当前实际布局：

```text
SpaceExp/
├── main_nav.py                 # 抽一组任务；配置独立，不从其他 main 导入
├── main_nav_batch.py           # 顺序跑完整个 benchmark，保存并支持续跑
├── generate_benchmark/
│   ├── generate_benchmark.py   # 帧筛选、配对、参考路线、两种目标图生成
│   ├── generate_train_trajectories_real_multi_step.py
│   ├── generate_train_trajectories_sim_single_step.py
│   ├── render_train_trajectories_real_multi_step.py
│   └── render_train_trajectories_sim_single_step.py
├── agent/
│   ├── loop.py                 # 模型调用、工具执行、图片预算、结束控制
│   ├── backend.py              # vLLM Chat Completions / OpenAI Responses
│   ├── models.py               # 模型工厂
│   ├── tools.py                # 相机工具和终止工具的 schema/实现
│   └── protocols/             # qwen / glm / deepseek / openai
├── scene/
│   ├── client.py               # 普通 Python 端，管理 Blender 和通信
│   ├── blender_worker.py       # Blender 请求循环
│   ├── renderer.py             # 点云/网格、相机、地面、运动
│   ├── boundary.py             # 外边界及高度限制
│   └── trajectory.py           # 俯视背景和轨迹图
└── backup/
    ├── main_agent.py
    ├── main_env.py
    └── generate_top_down_views.py
```

**交接时根目录没有 `main_agent.py`、`main_env.py`、`generate_top_down_views.py`；它们在 `backup/`。** README 中仍有旧入口说明。不要擅自搬回。备份脚本若要运行，应先检查相对导入和路径是否适合当前位置。

原 `agent/navigation.py` 先移到根目录，之后按用户要求彻底拆除：

- `merge_goal_tasks` 在 `generate_benchmark/generate_benchmark.py`。
- `select_navigation_tasks`、`navigation_goal` 分别存在两个导航入口里，入口相互独立。
- 根目录和 `agent/` 均没有 `navigation.py`。

最近检查到的配置（用户可能继续修改，以实际顶部配置为准）：

- `main_nav.py`：`MODEL='qwen3.8-flash-next'`，`GOAL_MODE='both'`；读取 `outputs/sub_benchmark_test.json`，固定随机种子 42 抽一对，同一对的两个模式分别新建环境执行。
- `main_nav_batch.py`：`MODEL='gpt-6.1-terra'`，`GOAL_MODE='top_down'`，同样读取 subset，`RESUME_DIR=None`。
- 两者 `max_steps=50`、图片预算 7、`preserve_prompt_images=True`、`verbose=True`。

## 4. 模型、历史和工具

`agent/protocols/__init__.py` 的 `MODEL_ADAPTERS` 显式映射模型到 backend/protocol：

- Qwen3.5-9B、Qwen3.8-27B、Qwen3.8-Flash-Next → vLLM / Qwen。
- GLM-5.3-Flash → vLLM / GLM。
- DeepSeek-V4.1-Flash → vLLM / DeepSeek。
- GPT-6 系列（含 Astra、6.1 Sol、6.1 Terra 等）→ OpenAI Responses。

每个模型自己的 `MODEL_GENERATION_CONFIGS[model]['server_config']` 包含地址和 timeout；不再使用全局 `MODEL_SERVER_CONFIG`。`create_model(model, model_config)` 只有两个参数。

常用本地端口：Qwen3.8 8000、Qwen3.5 8001、GLM 8002、DeepSeek 8003。不要假定当前服务仍在运行。

本地 `conversation.json` 保留图片路径，发送请求时才转成 base64。GPT 使用 `store=False`，本地保存并回传 `response_items` 中的加密推理状态，另存可读 reasoning summary；不依赖 `previous_response_id`。

**提示词和 tool message 均使用英文。** 不引入 coding、搜索、任意文件读取等通用 agent 工具。

当前工具开关都在各入口的 `TOOL_CONFIG`。默认开放 observe、finish_task、reset、前进及四种转向；后退、左右平移、垂直移动、top-down 观察工具目前关闭，但实现可选。

- `camera_action(key, magnitude)`：默认 move 0.3 米，turn 15 度，幅度必须正且有限。
- 水平移动按相机的水平 heading，俯仰不会让 forward 穿入地面；yaw 绕 scene up，pitch 保留 heading；没有 roll action。
- `camera_reset` 回 initial 完整位姿，不清除历史和步数；轨迹图不连接 reset 的瞬移线。
- `finish_task(status, reason)` 标记 terminal，立即结束 loop；其 status 是模型自报，不是独立指标。
- `max_steps` 数的是**模型调用轮数**，不是 camera action 次数；一轮可有多个工具调用。最后一轮动作会执行，仍未结束则 `status='failure', termination='max_steps'`。
- 每个兄弟任务有自己独立的 50 轮额度。批量每任务新 Agent、新 Blender；只复用无状态模型客户端。

## 5. 场景、数据路径和生成器

数据根目录：`/data4/haibo/data/3D_Env`。

场景格式：`<dataset>/<scene>/{reconstruction.ply,cameras.json,frames/,<scene>.mp4}`。

`train.json` / `test.json` 保存场景文件夹相对路径，不带 `.mp4`。ScanNet 的重复 scan 已按物理场景去重。测试集合有 ScanNet 142 个场景、ScanNet++ 50 个、ARKitScenes 150 个。

路径约定非常重要：

- `scene` 相对于 `DATA_ROOT`。
- `initial_view`、`target_view`、`reference_route` 相对于场景目录。
- `goal_image` **相对于 `DATA_ROOT`**，例如 `top_down_goals/scannetpp/6115eddb86/6115eddb86_0002.png`。
- 导航入口使用 **`DATA_ROOT / goal_image`**，不要改成相对于 benchmark JSON 的 `../../../data/...`。
- `preview` 相对于 benchmark JSON 所在目录。
- 俯视目标图位于 `/data4/haibo/data/3D_Env/top_down_goals/<dataset>/<scene>/<pair_id>.png`。

`generate_benchmark/generate_benchmark.py` 当前读取 `DATA_ROOT/'train.json'`，输出 `outputs/benchmark_train.json` 和训练预览。一次生成 first-person/top-down 两种任务，不再有独立的 `generate_top_down_goals.py` 或 `view_filter.py`。

核心选择规则：

- 每场景最多 6 对，软配额 near/medium/far 2/2/2，不强制凑齐；每对展开为两个任务，最多 12 条记录。
- 严格 `initial_frame_index < target_frame_index`，因为原视频中间轨迹将来要用作训练示范。
- 同一 initial 最多用于一对，同一 target 最多用一次；两种角色分别计算复用。
- 图片传统结构筛选（边缘、角点、4×4 覆盖、对比度等），只允许 Keep 作为端点；中间帧不丢弃。
- 按位置、朝向、pitch、高度分组去重，再软性增加端点分散度和降低路线网格重叠。
- 高度差 <=0.15 米，两端 roll 绝对值 <=5°；target 上仰最多 20°，两端绝对 pitch 还有 60° 约束。
- 近中远依据每场景候选距离的 1/3、2/3 分位数，不是全局固定距离阈值。
- 验证视频参考路线的位姿连续性和投影后的外边界，不检查内部家具碰撞；参考轨迹也不是最短路径。

既有测试 benchmark：

- `outputs/benchmark_test.json`：1860 对，3720 条记录。
- `outputs/sub_benchmark_test.json`：120 对，240 条记录；120 个不同场景；三个来源各 40 对，Near/Medium/Far 各 40 对。
- 上次训练生成日志：4317 候选场景，4115 accepted / 202 rejected；22523 对、45046 条任务。202 个被排除原因：98 无兼容配对、85 可用视角不足、19 无有效任务/路线。该次 `.report.json` 交接时已不在 outputs 根目录，不要假设仍存在。

原始 `results.json` / 单任务 `result.json` **目前没有 `distance_group`**。通过 `(scene,id)` 匹配批次内 `benchmark.json` 获取；单任务 `task.json` 中也有 `benchmark_task.distance_group`。分析 CSV 已含此字段。用户尚未要求修改生产代码自动写入它。

## 6. 渲染、俯视图及已知限制

- 地面由整段相机 up 方向和低点 RANSAC 联合估计，不固定世界 Z，也不只用第一帧 up。
- 初始相机使用指定视频帧的完整外参，包括 roll/pitch；渲染固定使用该 initial camera 的内参。
- 方形 960 渲染保留水平视场、扩展竖直视场，并非原视频的完全相同裁剪。
- 每个观察保存 PNG 和相机 JSON、`trajectory.json`，每步的 `*_top_down.png` 及最新 `top_down.png`。
- 历史轨迹无方向箭头；Start 绿色、Target 橙色、Current 红色，三者有朝向。
- 边界是点云外轮廓的二维网格，内缩 0.15 米；填充内部孔洞，保留外部凹口。高度限制默认离地 0.15..2.5 米。
- **没有完整的内部墙体/家具碰撞检测，且动作幅度没有固定上限。** 不能把当前指标等同于真实避障导航或把少轮数等同于最短安全路线。

纯俯视图脚本目前在 `backup/generate_top_down_views.py`，输出 `/data4/haibo/data/3D_Env/top_down_views/<dataset>/<scene>.png`：960×960、4 samples、1.8 米屋顶裁剪；无箭头、轨迹、轴、图例或边界线，默认覆盖。

已生成 5465 张：ScanNet 1513、ARKitScenes 3046、ScanNet++ 906。三个 ARKit 场景 `42445635`、`42898977`、`42899235` 的 camera-up 不一致导致失败；它们已从 train/test manifest 删除，但原场景文件夹仍在，因此遍历目录时仍会失败。不要盲目放宽地面方向检查或重新全量跑图。

曾发生 Blender 300 秒超时，发生在首次模型调用之前。日志一度为空或停在渲染采样完成后；同场景独立测试原参数约 11–12 秒可完成。曾观察到内存等待，与后台 DeepSeek 加载可能有关，**未证实具体根因，未据此修改生产代码**。

DeepSeek 本地 checkpoint 48 shards 合计约 475.25 GiB，其中 Engram 表及 scales 约 188.833 GiB。CPU offload 会使用主机内存。vLLM 当时环境版本 0.30.0；进度条 48/48 可能先完成权重遍历，之后才真正复制/处理权重。当前工具隔离环境看不到宿主机 GPU/进程，不能将工具内 `nvidia-smi` 失败视为宿主机 GPU 故障。

## 7. 四模型批次与固定评估规则

所有路径以下均以 `/data4/haibo/code/SpaceExp/outputs/` 为前缀：

| Model | Batch Directory |
|---|---|
| Qwen3.5-9B | `nav_batch_qwen3.5-9b_both_20261110_195144_619807` |
| Qwen3.8-27B | `nav_batch_qwen3.8-27b_both_20261111_021221_177781` |
| Qwen3.8-Flash-Next | `nav_batch_qwen3.8-flash-next_both_20261110_004126_552062` |
| GPT-6.1-Terra | `nav_batch_gpt-6.1-terra_both_20261115_114818_848481` |

每批 240 条，same 120 pairs。已核对 benchmark 快照、initial/target poses、agent/tool/scene/system-prompt 设置。Qwen 都开 thinking，输出上限 16384；Terra effort medium。Terra first-person 用较旧任务提示词模板，不能声称四组所有 prompt 完全一致。

当前用户确认的表格口径：

1. 只评价**最终**位置和方向。
2. 位置：`norm(final.position - target.position) <= 0.5`，完整三维距离。
3. First-person：归一化 forward 的夹角 <=30°，不比较完整 SO(3) roll。
4. Top-down：将保存的 `final.heading` 和 target.forward 投影到 scene_up 的地面平面，再比较夹角 <=30°。**用保存的 heading，避免镜头接近垂直时投影 forward 不稳定。**
5. `max_steps`、`error`、`interrupted` 终止计失败。模型自报的 success/failure 不是几何指标。
6. Steps 数 assistant/model 调用轮次，包含 finish_task，不含初始观察；生成截断/API 连接失败且未保存 assistant 的尝试要再加 1。

不要混用旧的仅按位姿达标的计数：Flash-Next first-person 有一条 `42446167_0002` 在第 50 轮最终位姿达标但因 max_steps 终止，**正式表记失败，因此是 14/120（11.67%），不是 15/120（12.50%）**。

Terra top-down 的 120/120 已独立读取源场景 `cameras.json` 和保存的 trajectory 复核，最差位置误差约 0.415 米、最差 heading 误差约 7.07°。不是只相信 summary 自报。110/120 条只前进一次，没有 reset；解读需结合前述无内部碰撞和大幅度动作的设置。

## 8. 记住这份 32 行完整分组表

用户希望以后反复展示的是：

`/data4/haibo/code/SpaceExp/outputs/navigation_four_models_both_0.5m_30deg_combined.md`

它是一张单表，列为 Model、Goal Type、Category、Success / Total、Success Rate、Mean Position Error (m)、Mean Direction Error (°)、Avg. Steps、Avg. Steps (Success)。按 first-person/top-down、Overall/Near/Medium/Far、四个模型展开为 **32 行**。每子类的所有误差、步数均独立统计，不只是 Overall。

其他正式分析产物（同目录、前缀 `navigation_four_models_both_0.5m_30deg`）：

- `.json`：完整数值、960 条逐任务指标、来源 SHA256 和统计口径。
- `.csv`：32 行分组统计。
- `.tasks.csv`：960 条逐任务数据，含 distance_group。
- `.md`：原两张分模态长表及诊断说明。
- `.png`、`.svg`：成功率和平均步数图。

旧的 `navigation_three_models_both_*`、`navigation_qwen9b_flash_terra_both_*` 是不同模型组合的中间报告；展示最新四模型表请用上述固定 combined 版本。

附录保存了相同 32 行数值，供此文单独阅读时使用。不要静默切换成功判据或覆盖这份固定结果。

## 9. Terra 合并目录已完成完全隔离

该 both 批次来自两个已完成的独立运行：

- first-person：`outputs/old/nav_batch_gpt-6.1-terra_20261108_124742_523625`
- top-down：`outputs/nav_batch_gpt-6.1-terra_top_down_20261115_114818_848481`

合并后按同一 pair 的 first-person/top-down 相邻排序，补齐 `goal_type`、`pair_id`、`direction_metric`，保留真实结果、相机参数、每条任务原始 prompt 和来源信息。

**用户拒绝硬链接，要求新旧目录完全隔离。已执行，不要再次改回硬链接/软链接：**

- 7592 个图片、日志、NPZ 文件均改为独立副本，额外约 6.19 GiB；JSON 本来就是独立副本。
- 新目录 11573 个文件，无软链接、无多重硬链接，与两个旧目录没有共享 inode。
- 所有任务 JSON 路径指向新目录；240 个 render.log 的旧路径也已更新。
- run.json/merge.json 的来源只保留历史批次名称和校验信息，不依赖旧目录路径。
- 已核验约 24840 个本地引用；两个旧目录保留，内容未变。移走旧输出目录不影响新输出目录查看和分析。
- 公共场景仍可引用 DATA_ROOT；这不属于对旧 output 的依赖。
- Terra 根目录及 240 个任务的 result.json 已按 Flash-Next 的字段顺序统一；`goal_type` 在首条记录第 6 行。递归字段结构、类型、数组形状、UTF-8/两空格格式都已核对一致。

`summary.json` 的 success rate 是**模型自报**，不等于上述评估成功率。Terra 合并自报 230/240；正式几何+步数结果是 210/240。

## 10. ScanNet 原始数据对照与 GS 讨论（尚未实施 GS）

原始 ScanNet 数据目录：`/data4/haibo/data/scannet-dataset/<scene>/`。本地 `_vh_clean_2.ply` 是带颜色的清理、降采样三角网格，`.labels.ply` 是语义着色，不要混用。当前 VGGT `reconstruction.ply` 只有点、没有面。

曾固定 seed 42 从 test 选厨房 `scene0011_00`、卫生间 `scene0664_00`、办公室 `scene0643_00`、卧室 `scene0382_00`，做原视频/ScanNet 顶点点渲染/VGGT 点渲染/ScanNet 原生 mesh 对比，各 3 个第一视角和俯视图。无本地官方 .sens 或 camera trajectory，使用全局 Sim(3) 对齐；厨房对齐可靠性低，不能当作严格同位姿误差评估。

定性结论：ScanNet 原生 mesh 表面通常更连续、结构更稳定；VGGT 点较密但可有孔洞、重复和局部变形。原始 mesh 顶点较稀，直接按点渲染可能颗粒更多，不能据此认为几何更差。

这些对照图原先在 `outputs/scannet_pointcloud_comparison_20261113/`，**交接时此目录已不在原位置**。不要宣称它还存在，若用户需要先定位移动位置或按需重做；不要为了交接重跑。

关于 Gaussian Splatting，仅完成方案讨论和官方文档核对，未安装、未训练、未接入：

- 可从 RGB 视频抽帧，经 COLMAP 位姿/稀疏点估计，再逐场景优化 GS。
- 推荐先考虑 gsplat 的标准 3DGS，再与 MCMC 对照；Splatfacto 是完整流程备选；2DGS 更关注表面几何和 mesh 提取。
- 每个成功注册的输入帧可导出 pose，但不保证原视频每一帧都能注册；单目结果还需解决米制尺度。
- GS 可导出 .ply，但其字段含中心、scale、rotation、opacity、SH 颜色，不是普通 RGB 点云或 mesh。gsplat 示例默认保存 .pt checkpoint，PLY 可选导出。
- 需要专用 GS rasterizer；现有 Blender 普通点渲染不能恢复 GS 外观。相机要另存、与 GS 保持相同坐标/尺度，若优化了 pose，应导出优化后的版本。
- GS 图像保真度不等于准确几何，不自动提供室内碰撞或可靠可行走面。

官方参考：https://docs.gsplat.studio/main/examples/colmap.html 、https://docs.nerf.studio/nerfology/methods/splat.html 、https://github.com/hbb1/2d-gaussian-splatting 。实际实施前重新核对版本。

## 11. 当前待办和安全继续方式

本次用户要求是写交接文档，其他最近任务已经完成。**没有授权继续安装 GS、训练、重跑 benchmark 或改动配额。** 新 session 应等待用户新的具体指示。

工作区有用户的未提交修改和移动（包括 `.gitignore`、README、入口、生成器、backup）。请先 `git status`，不要自动恢复。分析脚本有些只在 `/tmp/spaceexp-*.py`，不要依赖它们一定仍存在；正式结果以 outputs 中保存的 JSON/CSV/MD 为准。

现有 mock 检查覆盖了三种 GOAL_MODE、任务隔离、目标图路径、步数耗尽、正常终止、异常续跑及中断恢复。此前检查未额外调用付费模型。后续只验证本次改动所需内容，避免无关的全量运行。

## Appendix: saved 32-row result table

| Model | Goal Type | Category | Success / Total | Success Rate | Mean Position Error (m) | Mean Direction Error (°) | Avg. Steps | Avg. Steps (Success) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | First-person | Overall | 0/120 | 0.00% | 1.731 | 63.95 | 23.33 | N/A |
| Qwen3.8-27B | First-person | Overall | 14/120 | 11.67% | 1.550 | 49.38 | 24.11 | 15.21 |
| Qwen3.8-Flash-Next | First-person | Overall | 14/120 | 11.67% | 1.427 | 38.34 | 25.01 | 16.21 |
| GPT-6.1-Terra | First-person | Overall | 90/120 | 75.00% | 0.527 | 13.61 | 19.93 | 18.79 |
| Qwen3.5-9B | First-person | Near | 0/40 | 0.00% | 1.169 | 56.62 | 24.53 | N/A |
| Qwen3.8-27B | First-person | Near | 9/40 | 22.50% | 0.880 | 35.09 | 21.43 | 17.00 |
| Qwen3.8-Flash-Next | First-person | Near | 12/40 | 30.00% | 0.828 | 26.53 | 20.45 | 17.08 |
| GPT-6.1-Terra | First-person | Near | 32/40 | 80.00% | 0.355 | 12.37 | 17.35 | 16.28 |
| Qwen3.5-9B | First-person | Medium | 0/40 | 0.00% | 1.508 | 67.81 | 20.70 | N/A |
| Qwen3.8-27B | First-person | Medium | 4/40 | 10.00% | 1.397 | 43.60 | 23.70 | 13.25 |
| Qwen3.8-Flash-Next | First-person | Medium | 2/40 | 5.00% | 1.359 | 37.87 | 25.20 | 11.00 |
| GPT-6.1-Terra | First-person | Medium | 30/40 | 75.00% | 0.355 | 10.81 | 20.40 | 19.93 |
| Qwen3.5-9B | First-person | Far | 0/40 | 0.00% | 2.515 | 67.43 | 24.75 | N/A |
| Qwen3.8-27B | First-person | Far | 1/40 | 2.50% | 2.373 | 69.45 | 27.20 | 7.00 |
| Qwen3.8-Flash-Next | First-person | Far | 0/40 | 0.00% | 2.094 | 50.62 | 29.38 | N/A |
| GPT-6.1-Terra | First-person | Far | 28/40 | 70.00% | 0.872 | 17.65 | 22.03 | 20.43 |
| Qwen3.5-9B | Top-down | Overall | 0/120 | 0.00% | 1.852 | 93.78 | 14.98 | N/A |
| Qwen3.8-27B | Top-down | Overall | 17/120 | 14.17% | 1.423 | 67.17 | 36.52 | 19.53 |
| Qwen3.8-Flash-Next | Top-down | Overall | 59/120 | 49.17% | 0.740 | 29.19 | 18.89 | 14.51 |
| GPT-6.1-Terra | Top-down | Overall | 120/120 | 100.00% | 0.092 | 1.49 | 5.21 | 5.21 |
| Qwen3.5-9B | Top-down | Near | 0/40 | 0.00% | 1.526 | 79.94 | 15.00 | N/A |
| Qwen3.8-27B | Top-down | Near | 7/40 | 17.50% | 0.902 | 68.42 | 32.73 | 18.43 |
| Qwen3.8-Flash-Next | Top-down | Near | 24/40 | 60.00% | 0.562 | 27.15 | 14.68 | 13.46 |
| GPT-6.1-Terra | Top-down | Near | 40/40 | 100.00% | 0.074 | 1.57 | 5.00 | 5.00 |
| Qwen3.5-9B | Top-down | Medium | 0/40 | 0.00% | 1.584 | 90.10 | 16.10 | N/A |
| Qwen3.8-27B | Top-down | Medium | 8/40 | 20.00% | 1.288 | 54.89 | 34.08 | 17.88 |
| Qwen3.8-Flash-Next | Top-down | Medium | 24/40 | 60.00% | 0.610 | 26.40 | 18.53 | 14.21 |
| GPT-6.1-Terra | Top-down | Medium | 40/40 | 100.00% | 0.097 | 1.22 | 5.18 | 5.18 |
| Qwen3.5-9B | Top-down | Far | 0/40 | 0.00% | 2.445 | 111.30 | 13.85 | N/A |
| Qwen3.8-27B | Top-down | Far | 2/40 | 5.00% | 2.078 | 78.20 | 42.75 | 30.00 |
| Qwen3.8-Flash-Next | Top-down | Far | 11/40 | 27.50% | 1.049 | 34.01 | 23.48 | 17.45 |
| GPT-6.1-Terra | Top-down | Far | 40/40 | 100.00% | 0.105 | 1.69 | 5.45 | 5.45 |
