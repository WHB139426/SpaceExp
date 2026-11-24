# SpaceExp

SpaceExp 在 ARKitScenes、ScanNet 和 ScanNet++ 的点云重建场景中运行图像目标导航。Agent 根据起点和目标参考图调用相机工具，Blender 按请求返回第一视角观察。项目支持第一视角和俯视地图两种目标，以及单动作和批量动作两种调用方式。

当前已完成导航基线评估和两套配对的 SFT 轨迹数据。每套包含 22,437 条第一视角轨迹，可组成 `multi_step + render`、`multi_step + frame` 和 `single_step + render` 三种训练配置。Qwen3.5-9B 的 `multi_step + render` 训练已保存 `checkpoint-300`，SFT 后的导航评估待完成。

## 运行导航

在项目根目录运行导航入口。`main_nav.py` 随机选一组起终点，`main_nav_batch.py` 遍历所选目标类型的全部任务。两个入口当前都读取 `outputs/sub_benchmark_test.json`，默认 `GOAL_MODE = 'both'`。在对应文件顶部配置 `MODEL`、`GOAL_MODE`、`BENCHMARK_JSON` 和 `DATA_ROOT`。`MODEL_GENERATION_CONFIGS[MODEL]` 包含生成参数及服务地址，`AGENT_CONFIG` 控制步数和历史图片预算，`SCENE_CONFIG` 配置 Blender 路径及渲染参数。

```bash
python main_nav.py
python main_nav_batch.py
```

`main_nav.py` 当前选择 Qwen3.8-Flash-Next，需先启动对应的本地 vLLM 服务。`main_nav_batch.py` 当前选择 GPT-6.1-Terra，运行前设置 `OPENAI_API_KEY`。GPT 使用 Responses API，Qwen、GLM 和 DeepSeek 使用本地 vLLM Chat Completions。各模型的 backend 和 protocol 在 `agent/protocols/__init__.py` 注册，导航入口共用 `agent.models.create_model`、相机工具和 agent loop。`verbose=True` 打印每轮工具调用、可用的推理摘要和 token usage。

GLM 和 DeepSeek 的推理配置及工具协议参考 `../AgentHarness/protocols/`，仅迁入模型协议，共用现有相机工具、图像历史管理和 agent loop。
在对应导航入口选择 `MODEL`，通过该模型的 `server_config` 配置服务。

| Model | Protocol | Default Endpoint | Reasoning Effort |
|---|---|---|---|
| `glm-5.3-flash` | GLM tool-call tags | `http://127.0.0.1:8002/v1` | `low`, `high`, `max` |
| `deepseek-v4.1-flash` | DeepSeek DSML | `http://127.0.0.1:8003/v1` | Integer from 1 to 100; default 75 |

GLM 和 DeepSeek 的 `max_tokens` 默认均为 `16 * 1024`，可在各入口单独调整。
GLM 默认 `preserve_thinking=True`，协议传递 `clear_thinking=False`；DeepSeek 按原仓库使用
`thinking_mode="thinking"`，输出应包含结束推理的 `</think>`。它们的 `reasoning_effort` 含义与 Qwen/GPT 不同，分别配置。

GPT 使用 `store=False`，每轮由本地重传历史。每个 assistant 消息的 `response_items` 保存原始工具调用、消息 phase 和加密推理状态，供下一轮回传。`conversation.json` 也保存 `reasoning_summary`、`usage` 和 `response_id`。导航保留起点和目标参考图，将超出历史图片预算的较旧观察替换成文本。

起始帧和目标帧来自场景的 `frames/`，并在 `cameras.json` 中匹配相机记录。环境从起始帧的完整相机位姿启动，模型收到两张参考图和对应 `goal_type` 的英文提示。

批量导航运行 `python main_nav_batch.py`。该入口在文件顶部独立维护模型、agent、工具、场景、提示词、`DATA_ROOT`、`BENCHMARK_JSON`、`OUTPUT_DIR` 和 `RESUME_DIR` 配置，不导入其他 main 文件。默认按 JSON 顺序运行全部任务，tqdm 显示整体进度。整个批次复用模型客户端；每条任务新建 Agent 和 Blender 环境，使用该条 initial/target 的相机参数，聊天、相机状态和轨迹相互独立。单条运行异常会记录为 `termination="error"` 并继续下一条；步数耗尽记为失败，`finish_task` 保留模型自报结果。配置或模型客户端初始化错误会直接报错。

批量输出结构如下，每条任务的输出内容与单任务入口一致：

```text
outputs/nav_batch_<model>_<goal_mode>_<timestamp>/
├── run.json                  # Configuration snapshot; excludes API keys.
├── benchmark.json            # Exact input task list.
├── results.json              # Latest result for every attempted task, in benchmark order.
├── summary.json              # Finished/pending counts, outcomes, termination reasons and reported success rate.
└── tasks/<source>/<scene>/<task_id>_<model>_<timestamp>/
    ├── task.json
    ├── initial_view.jpg
    ├── target_view.jpg
    ├── conversation.json
    ├── trajectory.json
    ├── result.json            # Outcome and final camera parameters, including on errors when available.
    ├── top_down.png
    └── ...                   # Observations, per-step camera JSON/maps and Blender log.
```

每完成一条任务就原子更新结果与汇总。`summary.json` 的 `complete` 表示所有任务都已尝试结束，`self_reported_success_rate` 记录已结束任务中模型自报成功的比例。失败包括模型报告的 failure 和步数耗尽，运行异常及没有调用结束工具的回答另行计数。Ctrl+C 保存中断任务与已有进度后退出。将 `RESUME_DIR` 设为对应批次目录可续跑，跳过已有结果的任务（包括失败及 error），重新执行中断或未保存结果的任务，并保留旧尝试目录。续跑会检查任务清单和运行配置一致，还可从每条任务的 `result.json` 重建丢失的批量汇总。首次运行保持 `RESUME_DIR=None`。

## Benchmark 生成

当前 `outputs/benchmark_train.json` 有 22,523 组起终点，`outputs/benchmark_test.json` 有 1,860 组，`outputs/sub_benchmark_test.json` 有 120 组。每组均包含一个 `first_person` 任务和一个 `top_down` 任务。

运行 `python generate_benchmark/generate_benchmark.py` 一次完成图片筛选、起终点配对、俯视目标图生成和两种任务合并。配置位于文件顶部；当前配置读取 `train.json`，写入 `outputs/benchmark_train.json`，预览保存到 `outputs/benchmark_train_previews/`。改为测试集时同时设置 `TEST_SPLIT`、`OUTPUT_JSON`、`PREVIEW_DIR`。使用随机种子 42 打乱并遍历所选 split 中全部已重建场景，包括 ARKitScenes、ScanNet 和 ScanNet++，没有场景数量上限。每个场景需有 `cameras.json` 和 `reconstruction.ply`。进度条按已处理场景计数，包括被排除的场景；报告中的 `complete` 表示已遍历全部候选场景。每个场景最多生成 6 组起终点配对（添加俯视兄弟后每组有 2 个任务变体），同一个 initial 最多用于 1 组配对，同一个 target 最多用于 1 组配对；起点和目标分别计算复用次数，同一帧仍可在不同任务中分别充当起点和目标。有效任务不足 6 条时保留已有结果，不因数量不足跳过整个场景；完全没有有效任务的场景仍排除，并在报告中记录原因。按位置、朝向、俯仰和高度分组去重，过滤模糊图像及异常姿态，约束起终点高度差不超过 0.15 米，并尽量平衡近、中、远距离及目标方位。所有任务必须满足原视频的 `initial_frame_index < target_frame_index`，参考路线也只沿视频时间向前。每条参考路线由连续视频位姿投影到起点高度，在该起点对应的外边界网格中逐段检查；不检查家具碰撞，也不将参考路线当作最短路径。生成过程不调用重建或语言模型，使用 Blender 渲染每个场景的俯视图背景。

`MAX_TARGET_PITCH_DEGREES=20` 单独限制目标视角最多向上看 20°：俯仰角相对于估计地面计算，正值朝上、负值朝下。只过滤 target，initial 仍受原有 `MAX_ABS_PITCH_DEGREES=60` 限制；两者仍沿用侧倾、清晰度等其他筛选。每条任务额外记录 `target_pitch_degrees`，报告保存上限参数与角度约定。这是姿态过滤，不是天花板语义识别，不能保证画面中完全不出现天花板。新规则在重新运行生成脚本后生效。

`MAX_ABS_ROLL_DEGREES=5` 要求 initial 和 target 的左右侧倾均在 ±5° 内，相对于估计的场景竖直方向判断。超限视角在候选去重之前排除，两端角度分别保存为 `initial_roll_degrees` 和 `target_roll_degrees`。这是选择近乎水平的原始帧，不旋转图片或修改相机外参；原有两端侧倾差不超过 `MAX_ROLL_DIFFERENCE_DEGREES=10` 的约束继续保留。

候选图片在网格去重前执行结构筛选，initial 和 target 都只允许 **Keep**，**Review / Reject 均排除**。
筛选逻辑直接位于 `generate_benchmark/generate_benchmark.py`：亮度归一化、边缘与角点检测、4×4 网格结构覆盖及局部信息量检查。
阈值集中在 `generate_benchmark/generate_benchmark.py` 顶部的 `VIEW_FILTER_CONFIG`，默认至少 30 个角点、6/16 个有效网格，综合分数不低于 45，并检查主要为平行线的画面。
不合格图片不能成为起终点；参考路线中间经过的原视频帧仍保留。有效端点不足时，该场景可能生成更少任务或被跳过。
生成报告的 `view_filter.scenes` 和 `view_filter.totals` 记录每场景和全局的 Keep / Review / Reject 数量、读取失败数及原因；数量统计在姿态筛选后、网格去重前。


任务清单写入 `OUTPUT_JSON`（当前为 `outputs/benchmark_train.json`），`scene` 相对于数据根目录，`initial_view`、`target_view` 和 `reference_route` 中的图片路径相对于场景目录。三栏预览保存到 `PREVIEW_DIR`（当前为 `outputs/benchmark_train_previews/`），可从其中的 `index.md` 查看；每张预览包含起点图片、目标图片和俯视参考路线。任务中的 `preview` 路径相对于 benchmark JSON 所在目录。`OUTPUT_JSON` 对应的 `.report.json` 保存筛选参数、候选随机顺序、入选和排除原因，以及建议的成功阈值（位置 0.3 米、视线夹角 15°）；近、中、远各 2 条的配额只作为选择偏好，报告记录每场景实际任务数和分布。重新生成时移除该场景上一次多余的任务预览。`main_nav.py` 从 `BENCHMARK_JSON` 指向的新位置读取任务。benchmark 生成不会修改数据集文件或自动运行导航评估。

在网格候选基础上，任务选择还分别检查已选 initial 集合和 target 集合的连续位姿：地面距离小于 `VIEW_DEDUP_DISTANCE=0.5` 米、高度差小于 `VIEW_DEDUP_HEIGHT=0.15` 米、光学视线夹角小于 `VIEW_DEDUP_ANGLE_DEGREES=30` 度同时满足时，拒绝该角色的近重复视角；位置接近但朝向或高度明显不同仍可保留。同一帧跨起点/目标角色使用仍然允许。起点优先远离已选起点；目标评分在原距离配额优先级下，结合方位和转角覆盖，奖励远离已选目标的位置，并惩罚与已选参考路线重叠的配对。目标距离除以候选位置的地面最大间距归一化，权重为 `TARGET_SPREAD_WEIGHT`。路线采用独立的 `ROUTE_GRID_SIZE=0.5` 米网格，每段以不超过半格的间隔采样；重叠率为共享格子数除以两条路线较小的格子数，短路线被长路线完全包含时记为 1，权重为 `ROUTE_OVERLAP_WEIGHT`。这是软评分，不保证路线完全不重叠，也不是内部碰撞检测。任务额外保存选取当时的 `target_spread_m` 和 `max_route_overlap`；首条任务分别为 `null` 和 0。报告记录近重复筛除次数及最大路线重叠率。

场景目录包含 `reconstruction.ply`、`cameras.json` 和参考帧所在的 `frames/`。Blender 使用已有点云，起始相机的完整姿态和内参取自起始帧记录。观察采用 `resolution` 指定的方形画面，保留原水平视场并扩展竖直视场。

## 模型工具

基础工具为 `camera_observe()` 和 `camera_action(key, magnitude=None)`，每次成功调用都附上一张当前第一视角 PNG。

`main_nav.py` 和 `main_nav_batch.py` 默认开启 `camera_actions(actions)`，通过 `TOOL_CONFIG['enable_actions']` 控制。传入任意长度的非空动作列表，比如 `[{"key":"turn_left","magnitude":45},{"key":"move_forward","magnitude":1.0},{"key":"turn_down"}]`。每个动作以执行上一项后的相机位姿为起点，沿用单动作的开关、单位和默认幅度。全部处理完后返回最后一张第一视角图及每项的执行状态。受阻动作保持相机原位，后续动作继续执行。批量只增加一次观察，俯视轨迹仍包含每次移动经过的位置。续跑旧批次时，把 `enable_actions` 设为 `False` 以匹配旧配置。

可选工具 `finish_task(status, reason)` 用于主动结束任务，`status` 只能是 `success` 或 `failure`，`reason` 为简短英文依据。导航入口默认通过 `TOOL_CONFIG['enable_finish_task']` 开启此工具。有效调用会记录工具结果并立即结束本次 agent loop；无效参数返回工具错误，允许模型重试。同一轮在结束工具之后的其他调用会标记为未执行。终端输出和 `result.json` 记录工具返回的状态，例如：

```json
{"status": "success", "reason": "The current viewpoint matches the goal location and viewing direction.", "termination": "finish_task"}
```

`finish_task` 的成功或失败是模型自报结果。`max_steps` 按模型调用轮数计数，最后一轮的工具仍会执行并记录。若该轮有效调用 `finish_task`，则采用其结果。步数耗尽时任务记为 `status="failure"`、`termination="max_steps"`，并保存最后相机状态。未调用结束工具便输出最终文本时，`termination` 为 `model_response`。运行异常为 `error`，手动中断为 `interrupted`。这三种情况的 `status` 为 `null`。

导航的 `result.json` 还包含 `final_camera`，从最后一次成功保存的观察直接读取，不额外渲染。字段包括观察序号 `observation_id`、图片路径 `image`、世界坐标位置 `position`、镜头前方向 `forward`、镜头上方向 `up`、水平前进方向 `heading`、离地高度 `height_above_floor` 和渲染内参 `intrinsics`（分辨率、焦距、主点及像素比例）。纯观察、受阻动作及俯视图调用后的结束同样记录实际最终位姿。尚未产生任何观察便异常退出时，该字段为 `null`。这些参数只保存在本地结果中，不改变模型收到的工具结果。

单任务 `result.json`、批量每条任务的 `result.json` 和总表 `results.json` 均包含 `target_camera`。它保留 `cameras.json` 原始记录，包括 `extrinsic`、`intrinsic`、`W/H`、帧索引、时间与原图路径；另外保存目标图片绝对路径 `image`、世界坐标位置 `position`、单位视线 `forward`、单位镜头上方向 `up` 和 OpenCV 约定的 `camera_to_world`。`extrinsic_convention="world_to_camera_opencv"` 表明原始外参将世界坐标变换到相机坐标；`position/forward/up` 与 `final_camera` 使用相同世界坐标系，可直接用于位置与朝向误差计算。目标内参对应原始 target 图片，`final_camera.intrinsics` 对应渲染后的观察图片。尚未读取到目标相机就出错时，`target_camera` 为 `null`。续跑旧批次时，会从各任务的 `task.json` 补齐旧格式结果中的目标参数，不重跑模型或渲染。

两个导航入口各自通过 `TOOL_CONFIG` 控制工具和动作。`enable_observe`、`enable_action`、`enable_actions`、`enable_reset` 分别控制观察、单动作、批量动作和重置。`enable_move_forward/backward/left/right/up/down` 分别控制六种平移，`enable_turn_left/right/up/down` 分别控制四种转向。当前允许前进和四种转向，`enable_top_down` 默认关闭，`enable_finish_task` 默认开启。被禁用的工具和动作会从模型的工具 schema 移除。

可选工具 `camera_reset()` 无参数，通过 `enable_reset` 控制，两个导航入口默认开启。它恢复本次任务起始视角的完整位姿，并新增一条 `command="reset"` 的第一视角观察。俯视图在重置位置断开轨迹。若重置后的渲染失败，相机恢复重置前的位姿。

可选工具 `camera_top_down()` 返回最新俯视图，显示起点、历史轨迹、当前位置及朝向，以及已配置的目标位置和朝向。它不移动相机、不增加轨迹记录；尚未取得任何观察时需先取得观察。两个入口的 `TOOL_CONFIG['enable_top_down']` 当前均为 `False`，设为 `True` 即可注册该工具；关闭后不出现在模型的工具列表中。普通观察和动作仍只返回第一视角，俯视图仅在模型主动调用此工具时提供。返回本步固定保存的图片，后续动作不会覆盖历史消息里的地图。

- `move_forward` / `move_backward`：沿当前水平前进方向前进或后退。初始方向来自配置初始帧视线在地面上的投影。
- `move_left` / `move_right`：相对于当前水平前进方向向左或向右平移，不转头。四种平移均保持相对场景向上轴的高度、镜头姿态和水平朝向；左右转头后，四种移动方向随之改变。
- `move_up` / `move_down`：沿场景 `up` 或其反方向垂直升降，保持镜头姿态和水平朝向，不跟随镜头俯仰或侧倾。
- `turn_left` / `turn_right`：绕场景向上轴旋转，同时更新水平前进方向和镜头朝向，保持位置和俯仰。
- `turn_up` / `turn_down`：只调整镜头俯仰，保持位置和水平前进方向；俯仰限制在 -90° 至 90°。
- `magnitude`：六种平移距离单位为米（一 scene unit 为一米），转角单位为度；省略时平移使用 `move_step`（默认 0.3 米），转动使用 `look_step`（默认 15 度）。

低头和抬头不会改变任何平移的方向；水平平移保持高度，只有 `move_up/down` 改变高度。镜头垂直朝上或朝下时仍保留已有水平前进方向。若第一帧本身垂直朝上或朝下，则根据其相机右轴确定初始水平前进方向。初始镜头位姿不自动调平。环境默认启用场景外边界空气墙，不检测家具和内部障碍物。观察与动作都由模型按需调用，不提供通用读图、规划、状态或 undo 工具。

## 会话输出

每条导航任务保存 `conversation.json`、观察 PNG、对应相机元数据 JSON 和 Blender 的 `render.log`。单动作图片及配套 JSON、轨迹图的文件名包含动作和幅度，例如 `0002_move_forward_0.3.png`、`0002_move_forward_0.3.json`、`0002_move_forward_0.3_top_down.png`。被边界阻止的动作追加 `_blocked`，普通观察使用 `0001_observe.png` 这样的文件名。`top_down.png` 随轨迹更新。

导航输出保存在 `outputs/nav_<场景名>_<模型名>_<时间戳>/`，额外保存 `initial_view.jpg`、`target_view.jpg`（保留原图片扩展名）和 `task.json`。参考图直接复制原文件，`task.json` 保存对应相机记录供本地检查；目标相机矩阵不作为文本发送给模型；启用俯视图工具后，模型可以通过地图标记看到目标位置和朝向。

本地俯视图用蓝色渐变绘制历史轨迹，红色 X 和红色箭头表示当前位置及前进方向，绿色圆点表示起点。导航任务额外显示橙色目标星标及目标镜头视线在地面上的投影箭头；视线垂直地面时只显示位置。显示范围包含目标标记，当前位置和朝向绘制在最上层。

新会话的 `conversation.json` 在 `image_url.url` 中保存本地图片绝对路径。后端仅在发送请求时读取图片并转换为 Base64 data URL，保留聊天记录中的路径。导航参考图在清理较旧观察图片时始终保留。

`trajectory.json` 累计记录每次观察的位置、距估计地面的高度 `height_above_floor`、镜头视线 `forward`、水平前进方向 `heading`、相机上方向、动作和图像路径。`magnitude` 保存补全默认值后的动作幅度，也是文件名中的幅度；`applied_magnitude` 保存实际执行幅度，俯仰被 ±90° 上限截断时可能更小。被边界阻止时记录 `blocked=true`、英文 `reason` 和 `applied_magnitude=0`，位置和姿态保持原样，并保存一张当前观察；这些动作状态也通过工具结果返回给模型。普通观察的幅度为 `null`、`blocked=false`、`reason=null`。六种平移单位均为米，转动单位为度。原地转头、受阻动作和重复观察都会保留记录。图中红色箭头表示当前前进方向，历史轨迹只绘制连线，颜色条使用观察序号。默认观察仅发送第一视角图片；启用 `camera_top_down` 后可按需查看俯视图，相机位姿参数仍只保存在本地。

`SCENE_CONFIG` 的 `trajectory_resolution`、`trajectory_samples` 和 `trajectory_max_height` 分别控制俯视图背景的分辨率、采样数和地面以上的可见高度，用于裁掉天花板；`trajectory_dpi` 和 `trajectory_arrow_length` 控制图表输出 DPI 和朝向箭头长度。`up` 统一定义水平移动、左右转动和本地地图的向上方向；默认从场景估计，也可以手动覆盖，不改变初始相机位姿。每张相机元数据 JSON 中的 `top_down_frame` 保存投影坐标轴、原点、中心和范围。隐藏文件 `.top_down_background.png` 缓存背景，地图范围未变时不会重复渲染背景。

自动估计使用全部相机上方向的中位数作为竖直参考，并优先在点云高度较低的 20% 采样点中寻找地面平面；候选平面及最终拟合结果与参考方向的夹角均不超过 30°，同时检查相机和场景点是否位于其上方。若没有受足够点支持的平面，则保留多帧估计的竖直方向、近似地面高度，并在图中标注 `approximate floor`；相机方向本身缺乏稳定共识时直接报错，需手动配置 `up`。

`boundary_enforced=True` 默认启用移动限制。每次平移会逐格检查从当前位置到终点的完整线段，包含跨格拐角和沿网格线移动的情况；任一部分离开内缩区域就拒绝整步，不会跨越外部凹口后跳回场景内，也不会自动缩短动作或移动起点。转向和观察始终可用。若起点本身在内缩范围外，六种平移都会受阻，需检查边界参数或初始视点。

`SCENE_CONFIG` 的 `min_camera_height=0.15` 和 `max_camera_height=2.5` 限制相机中心距估计地面的高度，单位为米，必须有限且满足 `0 <= min < max`。启用 `boundary_enforced` 时，六种平移的整条路径都必须处于该高度区间内；垂直升降同时检查当前位置是否在水平允许区域。越界拒绝整步，不截短幅度，继续保存受阻观察和轨迹。初始位姿不会被调整；若其高度在区间外，平移会受阻，需调整配置，观察和转向仍可用。每张相机元数据 JSON 的 `height_limits` 保存上下限、地面参考点、竖直轴和启用状态。这些是配置的高度界限，并非自动检测的天花板，不检查局部低顶或家具。俯视图中纯升降的位置投影重合，实际高度变化记录在轨迹中。

`boundary_preview=True` 控制本地俯视图是否显示边界，关闭显示不会关闭移动限制。黄色线表示场景外轮廓，绿色线表示按 `camera_radius` 向内收缩后的允许移动范围。边界使用完整点云中距地面 -0.1 米至 `trajectory_max_height` 的点进行投影，填满所有封闭的内部空洞，保留与外部连通的凹口和 L 形结构，并选取与初始位置连通的区域；若起点不在轮廓内，则选取最近区域并标注起点异常。家具和内部障碍物不会因此变成碰撞体。

两个入口顶部的 `boundary_cell_size=0.05` 控制网格精度，`boundary_close_radius=0.15` 控制小缺口修补半径，`camera_radius=0.15` 控制内缩距离。设 `boundary_enforced=False` 会关闭水平和高度移动限制，保留按配置显示的边界预览。每个任务的 `boundary.npz` 保存外轮廓掩码、内缩掩码、坐标基、参数、起点标记和 `boundary_enforced` 状态，并用 `outer_only=True` 标明只使用外边界。

导航 agent 和轨迹绘图由普通 Python 运行，渲染端使用 Blender 自带的 `bpy` 和 NumPy。

## 俯视目标兄弟任务

`generate_benchmark/generate_benchmark.py` 为每组选中的 initial → target 自动生成两个任务变体，并写入同一个 JSON，
不再需要单独运行额外脚本。每个场景最多 6 对，即最多 12 条任务记录；配额和去重仍以配对为单位。
第一视角任务使用原 ID，兄弟任务的 `id` 添加 `_top_down`，两者共享 `pair_id`，用 `goal_type` 区分。
原始 `initial_view`、`target_view` 和参考路线继续保留。
俯视任务另外保存 `goal_image`、`direction_metric="horizontal_heading"` 和 `scene_up`；
第一视角任务保存 `direction_metric="forward_angle"`。`initial_view` 和 `target_view` 相对于场景目录；
`goal_image` 相对于 `DATA_ROOT`，例如 `top_down_goals/scannetpp/<scene>/<配对ID>.png`，
两个导航入口通过 `DATA_ROOT / goal_image` 读取；`preview` 相对于 benchmark JSON 所在目录。

图片保存到 `GOAL_IMAGE_DIR/<source>/<scene>/<配对ID>.png`，默认目录为 `/data4/haibo/data/3D_Env/top_down_goals/`。
分辨率、DPI、箭头长度和颜色由顶部 `GOAL_IMAGE_RESOLUTION`、`GOAL_DPI`、`GOAL_ARROW_LENGTH`、
`GOAL_START_COLOR`、`GOAL_TARGET_COLOR` 控制。默认 960×960，只显示静态场景背景、绿色 Start 和
橙色 Target 及两者的水平朝向箭头，不绘制参考路线、执行历史或目标第一视角。图例为英文，保留米制地面坐标尺度。
每个场景只渲染一次俯视背景，所有配对复用该背景叠加标记。

`PREVIEW_DIR/index.md` 同时链接第一视角配对预览和俯视目标图。报告中的 `goal_variants` 区分配对数、
任务总数和两种类型数量；每场景统计同时记录 `pairs`、`tasks`、`goal_types`。
重新生成时清理当前场景多余的配对预览和目标 PNG；生成失败或没有有效配对的场景也会清理其旧图。

两个导航入口都在顶部独立配置一个参数，无需切换 benchmark 文件：

```python
GOAL_MODE = 'first_person'  # 'first_person', 'top_down', or 'both'.
```

`main_nav_batch.py` 运行所选类型的全部任务；`both` 按每对先 first-person、后 top-down 的顺序运行。
`main_nav.py` 随机选择一个配对；`both` 连续运行该配对的两个版本。
每次都新建 Agent、Blender 环境和输出目录，恢复同一 initial pose，聊天和执行轨迹不共享。
单任务入口在 `both` 下即使第一个版本运行报错也会继续第二个版本，用户中断仍立即停止。
批量输出目录包含 `GOAL_MODE`，`run.json` 记录所选模式，`benchmark.json` 只保存实际选中的任务。
`summary.json` 保留总体统计，同时用 `by_goal_type` 分别统计两种任务，续跑会核对模式和任务清单。

两个入口根据每条任务的 `goal_type` 选择目标图片，并从顶部 `GOAL_PROMPTS` 只拼入对应提示。
旧任务缺少 `goal_type` 时按 `first_person` 处理；选择 `both` 时要求每组兄弟齐全且起终点一致。
模型仍只接收 initial 和 goal 两张参考图：first-person 任务发送原目标帧，top-down 任务发送地图。
本地额外保留 `target_view.*` 作 GT，地图另存为 `goal_image.png`，目标相机始终从原始 `target_view` 查找。
`task.json` 保存实际英文 prompt，结果保存 `goal_type`、`pair_id`、`direction_metric`，俯视任务还保存 `scene_up`。
后续客观评分应按 `direction_metric` 选择完整 forward 夹角或水平朝向夹角；summary 仍统计模型自报结果。

## First-person 导航基线

下表使用 `sub_benchmark_test.json` 中同一组 120 条第一视角任务。位姿成功要求最终三维位置误差 ≤ 0.5 m 且完整视线方向夹角 ≤ 30°，`max_steps`、`error` 和 `interrupted` 均计为失败。批次 `summary.json` 中的 `self_reported_success_rate` 另行记录模型自报结果。批量动作开启时，模型可以使用 `camera_action` 和 `camera_actions`。

| 模型 | 单动作工具 | 开启批量动作工具 |
|---|---:|---:|
| GPT-6.1-Terra | 90/120 (75.00%) | 74/120 (61.67%) |
| Qwen3.8-Flash-Next | 14/120 (11.67%) | 23/120 (19.17%) |
| Qwen3.8-27B | 14/120 (11.67%) | — |
| Qwen3.5-9B | 0/120 (0.00%) | 3/120 (2.50%) |

这些数字来自 `outputs/nav_batch_*` 下的历史批次。模型参数和提示词以各次 `run.json` 为准。Terra 开启批量动作的批次中有 22 条以 `error` 结束。

## Qwen3.5-9B 导航 SFT

### 轨迹数据

轨迹从 `benchmark_train.json` 中的第一视角 `reference_route` 和已记录相机位姿生成。两种生成器都保留起点和终点，目标抽取 5–10 个 waypoint。参考路线更短时保留所有点。训练用 JSON 都以 `camera_observe` 开始，以 `finish_task` 结束，并在 `route_actions` 中保存执行后的相机位姿。

| 轨迹 | 生成方式 | JSON |
|---|---|---|
| Multi-step | 随机选取中间 waypoint，按原视频顺序连接。每段用一组动作到达下一 waypoint 的地面位置，并匹配它的水平朝向和俯仰。沿途保持起始相机高度，沿用起始侧倾。 | `outputs/benchmark_train_traj_real_multi_step.json` |
| Single-step | 从参考路线按索引均匀采样。中间 waypoint 只约束地面位置，终点匹配目标视线。起始俯仰绝对值超过 30° 时先调平。后续前进至少 0.2 m 时，将 ≥120° 的水平转向均分成两步。前进 ≥1.6 m 时，均分成每步最多 1 m 的动作。 | `outputs/benchmark_train_traj_sim_single_step.json` |

清理后两份 JSON 各有 **22,437 条**轨迹，覆盖 **4,114 个**场景，起点和终点逐条对应。Near、Medium、Far 分别为 9,300、7,180、5,957 条。当前训练清单中的参考图和观测图路径均已核对。

| 轨迹 | Step 总数 | 每条平均 Step | Step 最小–最大 | Action 总数 | 每条平均 Action | Action 最小–最大 |
|---|---:|---:|---:|---:|---:|---:|
| Multi-step | 187,939 | 8.38 | 3–11 | 572,260 | 25.51 | 4–36 |
| Single-step | 412,411 | 18.38 | 6–42 | 367,537 | 16.38 | 4–40 |

训练数据中的 Step 按 `route_actions` 里的工具调用计数，包含 `camera_observe` 和 `finish_task`。Action 指一次前进或转向，俯仰也计为转向。当前 multi-step 数据的每个 `camera_actions` 包含 4 个 action，共返回一张观测图。包含两张参考图后，每条对话平均有 9.38 张图片（multi-step）或 19.38 张图片（single-step）。同一个 case 的三种配置可见本地预览 `outputs/analysis/paired-navigation-trajectories.jpg`。

### 生成与渲染

轨迹生成脚本的 `OUTPUT_JSON` 目前使用旧文件名 `benchmark_train_traj_frame.json` 和 `benchmark_train_traj.json`。准备新数据时，在对应脚本顶部设置 `BENCHMARK_JSON`、`DATA_ROOT` 和 `OUTPUT_JSON`，并将渲染脚本的 `TRAJECTORY_JSON` 指向生成的清单。生成器导出完整候选路线，当前训练清单是在渲染后筛除不完整轨迹，再按共同起终点对齐得到的。

```bash
python generate_benchmark/generate_train_trajectories_real_multi_step.py
python generate_benchmark/generate_train_trajectories_sim_single_step.py

OPENBLAS_NUM_THREADS=1 python generate_benchmark/render_train_trajectories_real_multi_step.py
OPENBLAS_NUM_THREADS=1 python generate_benchmark/render_train_trajectories_sim_single_step.py
```

渲染脚本用多个 Blender worker 回放动作、检查位姿并显示 `tqdm` 进度。`NUM_WORKERS` 和试跑上限 `MAX_TRAJECTORIES` 在脚本顶部设置。相同输入及渲染配置可重新运行，复用已成功且图片齐全的任务。脚本在完成或处理 `Ctrl+C` 时写出 `rendered.jsonl`，将成功轨迹的相对图片路径回写到输入 JSON。`SYNC_ONLY=True` 可利用已有结果同步图片路径。

### 启动训练

训练配置集中在 [sft/scripts/train.sh](sft/scripts/train.sh)。`data_root` 指向 `/data4/haibo/data/3D_Env`，两张参考图从其下的场景目录读取。按下表设置 `trajectory_type`、`observation_source`、`trajectory_json` 和 `observation_root`。表中 `$data_root` 代表上述目录，`trajectory_json` 在脚本中使用绝对路径。

| `trajectory_type` | `observation_source` | `trajectory_json`（`outputs/` 下） | `observation_root` | 观测字段 |
|---|---|---|---|---|
| `multi_step` | `render` | `benchmark_train_traj_real_multi_step.json` | `$data_root/train_traj_real_multi_step_renders` | `render_img_path` |
| `multi_step` | `frame` | `benchmark_train_traj_real_multi_step.json` | `$data_root` | `frame_path` |
| `single_step` | `render` | `benchmark_train_traj_sim_single_step.json` | `$data_root/train_traj_sim_single_step_renders` | `image_path` |

[NavigationTrajectoryDataset](sft/dataset/navigation.py) 按配置自动选择观测字段，把一条轨迹组织成一段多模态对话。动作幅度在编码时保留一位小数。每个 assistant 工具调用使用 Qwen3.5 的空 `<think>` 前缀，loss 只覆盖工具调用到回合结束 token。训练没有单独的验证集，也没有推理时的步数和历史图片预算。Collator 默认在 49,152 tokens 截断，并同步调整图像输入。

从项目根目录启动训练。依赖见 [requirements.txt](requirements.txt)，`flash-attn` 需与训练环境中的 PyTorch 和 CUDA 匹配。

```bash
bash sft/scripts/train.sh
```

脚本当前使用 `multi_step + render`、3 张 GPU、每卡 batch size 4、梯度累积 2、学习率 `2e-5` 和 1 个 epoch。训练使用 BF16、FlashAttention 2、DeepSpeed ZeRO-3 和 gradient checkpointing，默认更新语言模型及 LM head，冻结视觉编码器和 merger。日志上报到 W&B 项目 `SpaceExp`，checkpoint 保存在 `outputs/sft/Qwen3.5-9B-nav-<trajectory_type>-<observation_source>/`。再次启动同一配置时，训练入口会自动查找该目录下最近的 checkpoint。

下面的命令可检查样本的对话文本、图片路径、张量形状和监督标签。它使用 `navigation.py` 当前的 `single_step + render` 调试配置，也可通过命令行指定其他数据模式。

```bash
python sft/dataset/navigation.py --num-samples 1
```

## vLLM 启动命令

### Qwen3.5-9B

使用本地权重和原生 262,144-token 上下文，不修改 RoPE。需要 vLLM >= 0.17.0；
在安装了 vLLM 的推理环境中运行。GPU 编号和显存参数按实际资源调整，先使用单并发。
参考 [vLLM 官方 recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B)。

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_ENGINE_READY_TIMEOUT_S=3600 \
vllm serve /data4/haibo/weights/Qwen3.5-9B \
  --served-model-name qwen3.5-9b \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.50 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":100,"video":0}'
```

启动后，在 `main_nav.py` 或 `main_nav_batch.py` 中将
`MODEL = "qwen3.5-9b"`，并确认对应 `MODEL_GENERATION_CONFIGS[MODEL]['server_config']['base_url']`
与服务地址一致。Qwen3.5-9B 默认使用 `http://127.0.0.1:8001/v1`，其他本地模型仍使用 8000 端口。
模型通过注册字典自动选择 `VLLMBackend + QwenProtocol`。

Qwen3.5 的配置只传 `enable_thinking`；本地模板不读取 `reasoning_effort` 或
`preserve_thinking`，因此不填写这两个选项。旧 Qwen 模型仍按各自配置传递这些参数。
工具调用由框架解析原始 XML 文本，启动时不要添加 `--enable-auto-tool-choice` 或
`--tool-call-parser`，客户端继续发送工具 schema 和 `tool_choice="none"`。

### Qwen3.8-Flash-Next-FP8 (vLLM 0.31.0)

Uses the native 262,144-token context window without RoPE scaling overrides.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_ENGINE_READY_TIMEOUT_S=3600 \
VLLM_PLE_CPU_OFFLOAD=1 \
vllm serve /data4/haibo/weights/Qwen3.8-Flash-Next-FP8 \
  --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --allowed-local-media-path /data4/haibo \
  --limit-mm-per-prompt '{"image":100,"video":100}'
```

### Qwen3.8-27B-FP8 (vLLM 0.31.0)

Requires `transformers>=5.8.0`. Uses the native 262,144-token context window
without RoPE scaling overrides.

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_ENGINE_READY_TIMEOUT_S=3600 \
vllm serve /data4/haibo/weights/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --allowed-local-media-path /data4/haibo \
  --limit-mm-per-prompt '{"image":100,"video":100}'
```

以下两个模型的命令参考 [AgentHarness/SERVING.md](../AgentHarness/SERVING.md)，原仓库使用 vLLM 0.30.0。
端口分别调整为 8002 和 8003，GPU 编号、并行度和显存配置按实际可用硬件调整。
客户端解析原生工具文本，因此两个服务也不启用自动工具解析。
DeepSeek 保留原仓库的 `deepseek_v41` tokenizer 和 Engram CPU offload 参数。

### GLM-5.3-Flash

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_ENGINE_READY_TIMEOUT_S=3600 \
vllm serve /data4/haibo/weights/GLM-5.3-Flash \
  --served-model-name glm-5.3-flash \
  --host 127.0.0.1 \
  --port 8002 \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 1048576 \
  --allowed-local-media-path /data4/haibo \
  --limit-mm-per-prompt '{"image":100,"video":100}' \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --attention-backend FLASH_ATTN_MLA_SPARSE
```

### DeepSeek-V4.1-Flash

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_ENGINE_READY_TIMEOUT_S=3600 \
vllm serve /data4/haibo/weights/DeepSeek-V4.1-Flash-1017 \
  --served-model-name deepseek-v4.1-flash \
  --host 127.0.0.1 \
  --port 8003 \
  --tensor-parallel-size 4 \
  --tokenizer-mode deepseek_v41 \
  --engram-config '{"cpu_offload":true}' \
  --kv-cache-dtype fp8 \
  --max-model-len 1048576 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --enforce-eager \
  --allowed-local-media-path /data4/haibo \
  --limit-mm-per-prompt '{"image":100}'
```
