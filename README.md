# SpaceExp

从现有点云场景启动一个可探索的第一视角环境。普通 Python 运行 agent，后台常驻一个 Blender 进程，按工具请求渲染。

## 运行

在对应入口 `main_agent.py` 或 `main_nav.py` 顶部选择 `MODEL`，并通过 `MODEL_GENERATION_CONFIGS[MODEL]` 设置该模型的生成与 thinking 参数。`MODEL_SERVER_CONFIG` 按 `vllm` / `openai` 分别配置服务地址和超时，`AGENT_CONFIG` 配置工具循环上限、历史图片预算和 `verbose` 输出，`SCENE_CONFIG` 集中配置场景目录、Blender 路径、渲染参数和步长。`agent/protocols/__init__.py` 的 `MODEL_ADAPTERS` 字典显式指定每个模型的 backend 和 protocol，两个 agent 入口共用 `agent.models.create_model` 工厂。

GPT-6 使用 `MODEL = "gpt-6-astra"`，然后设置环境变量并启动：

```bash
export OPENAI_API_KEY="your_api_key_here"
python main_agent.py
# Or: python main_nav.py
```

GPT 使用官方 Responses API；Qwen 继续使用本地 vLLM Chat Completions 和 Qwen 文本工具协议。GPT 默认 `reasoning_effort="high"`、`reasoning_summary="auto"`，不传 Qwen 的采样和模板参数。所有模型共用相机工具、动作开关、场景渲染和 agent loop。GPT 每轮自动选择工具或结束回答，最多调用一个工具；不会强制连续移动。`verbose=True` 打印每轮工具调用、GPT 推理摘要（若有）和 token usage；摘要不是原始 CoT。

GPT 使用 `store=False`，每轮由本地重传历史，不使用 `previous_response_id` 或服务器会话。每个 assistant 消息的 `response_items` 保存完整原生输出，包括原始工具调用、消息 phase 和不可读的加密推理状态；下一轮原样回传，`reasoning_summary` 仅供阅读。`usage` 和 `response_id` 也记录在 `conversation.json`。图片历史仍采用现有预算；导航保留 initial/target，较旧观察图片替换为文字，发送请求时才嵌入仍保留的图片。`/new` 清空这些历史和推理状态，旧会话文件保持不变。图片被移除后，已有文字和推理状态仍可能保留对旧图片的理解。

手动探索使用 `python main_env.py`，无需模型服务。场景和渲染参数在该文件顶部的 `SCENE_CONFIG` 中独立配置。启动后自动保存初始观察，随后每行输入一个动作，例如 `move_forward 0.3`、`turn_left 30`、`turn_up 15`；省略幅度则使用配置的默认步长。`observe` 再次观察，`help` 显示命令，`/new` 在新目录重置环境，`/exit` 退出。输出保存在 `outputs/env_<场景名>_<时间戳>/`；终端显示当前第一视角图片和最新俯视轨迹图的路径，打开保存的 PNG 即可查看。

非交互式图像目标导航使用 `python main_nav.py`。在文件顶部配置 `INITIAL_VIEW`、`GOAL_VIEW` 和英文 `PROMPT`；两张图片必须来自场景的 `frames/` 且能在 `cameras.json` 中匹配。环境从 initial view 的完整相机位姿启动，模型接收两张参考图及任务文本。`MODEL`、模型生成参数、服务地址、输出目录、系统提示词以及 agent、场景和工具配置均在 `main_nav.py` 内独立定义，不从 `main_agent.py` 导入。两个入口的配置分别维护。

批量导航运行 `python main_nav_bacth.py`（文件名沿用 `bacth`）。该入口在文件顶部独立维护模型、agent、工具、场景、提示词、`DATA_ROOT`、`BENCHMARK_JSON`、`OUTPUT_DIR` 和 `RESUME_DIR` 配置，不导入其他 main 文件。默认按 JSON 顺序运行全部任务，tqdm 显示整体进度。整个批次复用模型客户端；每条任务新建 Agent 和 Blender 环境，使用该条 initial/target 的相机参数，聊天、相机状态和轨迹相互独立。单条运行异常会记录为 `termination="error"` 并继续下一条；步数耗尽记为失败，`finish_task` 保留模型自报结果。配置或模型客户端初始化错误会直接报错。

批量输出结构如下，每条任务的输出内容与单任务入口一致：

```text
outputs/nav_batch_<model>_<timestamp>/
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

每完成一条任务就原子更新结果与汇总；`summary.json` 的 `complete` 表示所有任务都已尝试结束，不代表全部成功。`self_reported_success_rate` 是已结束任务中模型自报成功的比例，尚未加入独立位姿评估；失败包括模型报告的 failure 和步数耗尽，运行异常及没有调用结束工具的回答另行计数。Ctrl+C 保存中断任务与已有进度后退出。将 `RESUME_DIR` 设为对应批次目录可续跑：跳过已有结果的任务（包括失败及 error），重新执行中断或未保存结果的任务，并保留旧尝试目录。续跑会检查任务清单和运行配置一致，还可从每条任务的 `result.json` 重建丢失的批量汇总。首次运行保持 `RESUME_DIR=None`。

运行 `python generate_benchmark.py` 可生成测试 benchmark。配置位于文件顶部：默认用随机种子 42 打乱并遍历 `test.json` 中全部已重建场景，包括 ARKitScenes、ScanNet 和 ScanNet++，没有场景数量上限。每个场景需有 `cameras.json` 和 `reconstruction.ply`。进度条按已处理场景计数，包括被排除的场景；报告中的 `complete` 表示已遍历全部候选场景。每个场景最多生成 6 条任务，同一个 initial 最多生成 1 条，同一个 target 最多使用 1 次；起点和目标分别计算复用次数，同一帧仍可在不同任务中分别充当起点和目标。有效任务不足 6 条时保留已有结果，不因数量不足跳过整个场景；完全没有有效任务的场景仍排除，并在报告中记录原因。按位置、朝向、俯仰和高度分组去重，过滤模糊图像及异常姿态，约束起终点高度差不超过 0.15 米，并尽量平衡近、中、远距离及目标方位。所有任务必须满足原视频的 `initial_frame_index < target_frame_index`，参考路线也只沿视频时间向前。每条参考路线由连续视频位姿投影到起点高度，在该起点对应的外边界网格中逐段检查；不检查家具碰撞，也不将参考路线当作最短路径。生成过程不调用重建或语言模型，使用 Blender 渲染每个场景的俯视图背景。

`MAX_TARGET_PITCH_DEGREES=20` 单独限制目标视角最多向上看 20°：俯仰角相对于估计地面计算，正值朝上、负值朝下。只过滤 target，initial 仍受原有 `MAX_ABS_PITCH_DEGREES=60` 限制；两者仍沿用侧倾、清晰度等其他筛选。每条任务额外记录 `target_pitch_degrees`，报告保存上限参数与角度约定。这是姿态过滤，不是天花板语义识别，不能保证画面中完全不出现天花板。新规则在重新运行生成脚本后生效。

`MAX_ABS_ROLL_DEGREES=5` 要求 initial 和 target 的左右侧倾均在 ±5° 内，相对于估计的场景竖直方向判断。超限视角在候选去重之前排除，两端角度分别保存为 `initial_roll_degrees` 和 `target_roll_degrees`。这是选择近乎水平的原始帧，不旋转图片或修改相机外参；原有两端侧倾差不超过 `MAX_ROLL_DIFFERENCE_DEGREES=10` 的约束继续保留。

任务清单写入 `outputs/benchmark_test.json`，`scene` 相对于数据根目录，`initial_view`、`target_view` 和 `reference_route` 中的图片路径相对于场景目录。三栏预览保存到 `outputs/benchmark_test_previews/`，可从其中的 `index.md` 查看；每张预览包含起点图片、目标图片和俯视参考路线。任务中的 `preview` 路径相对于 benchmark JSON 所在目录。`outputs/benchmark_test.report.json` 保存筛选参数、候选随机顺序、入选和排除原因，以及建议的成功阈值（位置 0.3 米、视线夹角 15°）；近、中、远各 2 条的配额只作为选择偏好，报告记录每场景实际任务数和分布。重新生成时移除该场景上一次多余的任务预览。`main_nav.py` 从 `BENCHMARK_JSON` 指向的新位置读取任务。benchmark 生成不会修改数据集文件或自动运行导航评估。

在网格候选基础上，任务选择还分别检查已选 initial 集合和 target 集合的连续位姿：地面距离小于 `VIEW_DEDUP_DISTANCE=0.5` 米、高度差小于 `VIEW_DEDUP_HEIGHT=0.15` 米、光学视线夹角小于 `VIEW_DEDUP_ANGLE_DEGREES=30` 度同时满足时，拒绝该角色的近重复视角；位置接近但朝向或高度明显不同仍可保留。同一帧跨起点/目标角色使用仍然允许。起点优先远离已选起点；目标评分在原距离配额优先级下，结合方位和转角覆盖，奖励远离已选目标的位置，并惩罚与已选参考路线重叠的配对。目标距离除以候选位置的地面最大间距归一化，权重为 `TARGET_SPREAD_WEIGHT`。路线采用独立的 `ROUTE_GRID_SIZE=0.5` 米网格，每段以不超过半格的间隔采样；重叠率为共享格子数除以两条路线较小的格子数，短路线被长路线完全包含时记为 1，权重为 `ROUTE_OVERLAP_WEIGHT`。这是软评分，不保证路线完全不重叠，也不是内部碰撞检测。任务额外保存选取当时的 `target_spread_m` 和 `max_route_overlap`；首条任务分别为 `null` 和 0。报告记录近重复筛除次数及最大路线重叠率。

场景目录需要 `reconstruction.ply` 和 `cameras.json`；无需原视频或 `frames/`，不加载重建模型。初始位置、完整姿态（包括俯仰和侧倾）及内参使用第一帧的记录；仅将 OpenCV 相机坐标转换为 Blender 相机坐标，不调平、不修改眼高。渲染继续采用 `resolution` 指定的方形画面，保留原水平视场并扩展竖直视场。

## 模型工具

基础工具为 `camera_observe()` 和 `camera_action(key, magnitude=None)`，每次成功调用都附上一张当前第一视角 PNG。

可选工具 `finish_task(status, reason)` 用于主动结束任务，`status` 只能是 `success` 或 `failure`，`reason` 为简短英文依据。通过 `TOOL_CONFIG['enable_finish_task']` 控制，导航入口默认开启，交互入口默认关闭。有效调用会记录工具结果并立即结束本次 agent loop，不再请求模型；无效参数返回工具错误，允许模型重试。同一轮在结束工具之后的其他调用会标记为未执行。终端输出固定格式 JSON，导航输出目录另存 `result.json`，例如：

```json
{"status": "success", "reason": "The current viewpoint matches the goal location and viewing direction.", "termination": "finish_task"}
```

`finish_task` 的成功或失败是模型自报结果，尚未进行独立位姿评估。耗尽 `max_steps` 仍未结束时，程序将任务记为 `status="failure"`、`termination="max_steps"`，正常结束循环并保存最后相机状态，不再抛步数超限异常。`max_steps` 按模型调用轮数计数；最后一轮的工具仍会执行并记录，若该轮有效调用 `finish_task`，则优先采用其结果。未调用结束工具便输出最终文本时，`termination` 为 `model_response`；运行异常为 `error`，手动中断为 `interrupted`，这三种情况的 `status` 仍为 `null`。程序异常仍保留原有报错行为。

导航的 `result.json` 还包含 `final_camera`，从最后一次成功保存的观察直接读取，不额外渲染。字段包括观察序号 `observation_id`、图片路径 `image`、世界坐标位置 `position`、镜头前方向 `forward`、镜头上方向 `up`、水平前进方向 `heading`、离地高度 `height_above_floor` 和渲染内参 `intrinsics`（分辨率、焦距、主点及像素比例）。纯观察、受阻动作及俯视图调用后的结束同样记录实际最终位姿。尚未产生任何观察便异常退出时，该字段为 `null`。这些参数只保存在本地结果中，不改变模型收到的工具结果。

单任务 `result.json`、批量每条任务的 `result.json` 和总表 `results.json` 均包含 `target_camera`。它保留 `cameras.json` 原始记录，包括 `extrinsic`、`intrinsic`、`W/H`、帧索引、时间与原图路径；另外保存目标图片绝对路径 `image`、世界坐标位置 `position`、单位视线 `forward`、单位镜头上方向 `up` 和 OpenCV 约定的 `camera_to_world`。`extrinsic_convention="world_to_camera_opencv"` 表明原始外参将世界坐标变换到相机坐标；`position/forward/up` 与 `final_camera` 使用相同世界坐标系，可直接用于位置与朝向误差计算。目标内参对应原始 target 图片，`final_camera.intrinsics` 对应渲染后的观察图片。尚未读取到目标相机就出错时，`target_camera` 为 `null`。续跑旧批次时，会从各任务的 `task.json` 补齐旧格式结果中的目标参数，不重跑模型或渲染。

`main_agent.py` 和 `main_nav.py` 顶部的 `TOOL_CONFIG` 显式列出全部 15 个布尔开关：`enable_observe`、`enable_action`、`enable_reset` 分别控制 `camera_observe`、`camera_action` 和 `camera_reset`；`enable_move_forward/backward/left/right/up/down` 分别控制六种平移；`enable_turn_left/right/up/down` 分别控制四种转向；`enable_top_down` 和 `enable_finish_task` 控制俯视图与结束工具。两个入口分别独立定义完整的 `TOOL_CONFIG`，互不继承，便于单独调整。关闭工具会将其从注册列表移除，关闭动作会将其从 schema 和描述移除，模型即使尝试调用也会被拒绝；全部动作关闭时，`camera_action` 自动不注册。交互入口当前开启所有移动、转向、观察与重置，关闭俯视图与结束工具；导航入口当前开启前进、四种转向、观察、重置和结束工具。这些开关只限制模型工具，手动入口 `main_env.py` 仍支持全部动作。

可选工具 `camera_reset()` 无参数，通过 `enable_reset` 控制，两个 agent 入口默认开启。它将相机直接恢复到本次任务 initial view 的完整位置、镜头姿态（含俯仰和侧倾）及水平朝向，并附上重置后的第一视角；导航中使用选中的 initial view，不一定是视频第一帧。目标、聊天记录、已有轨迹、会话目录和剩余步数均保留，新增一条 `command="reset"` 的观察。俯视图保留历史路线，但不连接重置的瞬移线段。重置不受移动边界阻止；若渲染失败则恢复重置前位姿。手动入口可输入 `reset`，区别于新建目录并清空轨迹的 `/new`。

可选工具 `camera_top_down()` 返回最新俯视图，显示起点、历史轨迹、当前位置及朝向，以及已配置的目标位置和朝向。它不移动相机、不增加轨迹记录；尚未取得任何观察时需先取得观察。两个入口的 `TOOL_CONFIG['enable_top_down']` 当前均为 `False`，设为 `True` 即可注册该工具；关闭后不出现在模型的工具列表中。普通观察和动作仍只返回第一视角，俯视图仅在模型主动调用此工具时提供。返回本步固定保存的图片，后续动作不会覆盖历史消息里的地图。

- `move_forward` / `move_backward`：沿当前水平前进方向前进或后退。初始方向来自配置初始帧视线在地面上的投影。
- `move_left` / `move_right`：相对于当前水平前进方向向左或向右平移，不转头。四种平移均保持相对场景向上轴的高度、镜头姿态和水平朝向；左右转头后，四种移动方向随之改变。
- `move_up` / `move_down`：沿场景 `up` 或其反方向垂直升降，保持镜头姿态和水平朝向，不跟随镜头俯仰或侧倾。
- `turn_left` / `turn_right`：绕场景向上轴旋转，同时更新水平前进方向和镜头朝向，保持位置和俯仰。
- `turn_up` / `turn_down`：只调整镜头俯仰，保持位置和水平前进方向；俯仰限制在 -90° 至 90°。
- `magnitude`：六种平移距离单位为米（一 scene unit 为一米），转角单位为度；省略时平移使用 `move_step`（默认 0.3 米），转动使用 `look_step`（默认 15 度）。

低头和抬头不会改变任何平移的方向；水平平移保持高度，只有 `move_up/down` 改变高度。镜头垂直朝上或朝下时仍保留已有水平前进方向。若第一帧本身垂直朝上或朝下，则根据其相机右轴确定初始水平前进方向。初始镜头位姿不自动调平。环境默认启用场景外边界空气墙，不检测家具和内部障碍物。观察与动作都由模型按需调用，不提供通用读图、规划、状态或 undo 工具。

## 会话输出

Agent 每次启动在 `outputs/chat_<场景名>_<模型名>_<时间戳>/` 保存 `conversation.json`、观察 PNG、对应相机元数据 JSON 和 Blender 的 `render.log`。动作图片及配套 JSON、轨迹图的文件名包含动作和幅度，例如 `0002_move_forward_0.3.png`、`0002_move_forward_0.3.json`、`0002_move_forward_0.3_top_down.png`；被边界阻止的动作追加 `_blocked`，例如 `0003_move_forward_100_blocked.png`。普通观察仍为 `0001_observe.png`。另外覆盖更新一张 `top_down.png` 方便查看最新状态。两个入口使用同一套渲染与记录逻辑。

导航输出保存在 `outputs/nav_<场景名>_<模型名>_<时间戳>/`，额外保存 `initial_view.jpg`、`target_view.jpg`（保留原图片扩展名）和 `task.json`。参考图直接复制原文件，`task.json` 保存对应相机记录供本地检查；目标相机矩阵不作为文本发送给模型；启用俯视图工具后，模型可以通过地图标记看到目标位置和朝向。

本地俯视图用蓝色渐变绘制历史轨迹，红色 X 和红色箭头表示当前位置及前进方向，绿色圆点表示起点。导航任务额外显示橙色目标星标及目标镜头视线在地面上的投影箭头；视线垂直地面时只显示位置。显示范围包含目标标记，当前位置和朝向绘制在最上层。

新会话的 `conversation.json` 在 `image_url.url` 中保存本地图片绝对路径。后端仅在发送请求时读取图片并转换为 Base64 data URL，保留聊天记录中的路径。导航参考图在清理较旧观察图片时始终保留。

`trajectory.json` 累计记录每次观察的位置、距估计地面的高度 `height_above_floor`、镜头视线 `forward`、水平前进方向 `heading`、相机上方向、动作和图像路径。`magnitude` 保存补全默认值后的动作幅度，也是文件名中的幅度；`applied_magnitude` 保存实际执行幅度，俯仰被 ±90° 上限截断时可能更小。被边界阻止时记录 `blocked=true`、英文 `reason` 和 `applied_magnitude=0`，位置和姿态保持原样，并保存一张当前观察；这些动作状态也通过工具结果返回给模型。普通观察的幅度为 `null`、`blocked=false`、`reason=null`。六种平移单位均为米，转动单位为度。原地转头、受阻动作和重复观察都会保留记录。图中红色箭头表示当前前进方向，历史轨迹只绘制连线，颜色条使用观察序号。默认观察仅发送第一视角图片；启用 `camera_top_down` 后可按需查看俯视图，相机位姿参数仍只保存在本地。

`SCENE_CONFIG` 的 `trajectory_resolution`、`trajectory_samples` 和 `trajectory_max_height` 分别控制俯视图背景的分辨率、采样数和地面以上的可见高度，用于裁掉天花板；`trajectory_dpi` 和 `trajectory_arrow_length` 控制图表输出 DPI 和朝向箭头长度。`up` 统一定义水平移动、左右转动和本地地图的向上方向；默认从场景估计，也可以手动覆盖，不改变初始相机位姿。每张相机元数据 JSON 中的 `top_down_frame` 保存投影坐标轴、原点、中心和范围。隐藏文件 `.top_down_background.png` 缓存背景，地图范围未变时不会重复渲染背景。

自动估计使用全部相机上方向的中位数作为竖直参考，并优先在点云高度较低的 20% 采样点中寻找地面平面；候选平面及最终拟合结果与参考方向的夹角均不超过 30°，同时检查相机和场景点是否位于其上方。若没有受足够点支持的平面，则保留多帧估计的竖直方向、近似地面高度，并在图中标注 `approximate floor`；相机方向本身缺乏稳定共识时直接报错，需手动配置 `up`。

`boundary_enforced=True` 默认启用移动限制。每次平移会逐格检查从当前位置到终点的完整线段，包含跨格拐角和沿网格线移动的情况；任一部分离开内缩区域就拒绝整步，不会跨越外部凹口后跳回场景内，也不会自动缩短动作或移动起点。转向和观察始终可用。若起点本身在内缩范围外，六种平移都会受阻，需检查边界参数或初始视点。

`SCENE_CONFIG` 的 `min_camera_height=0.15` 和 `max_camera_height=2.5` 限制相机中心距估计地面的高度，单位为米，必须有限且满足 `0 <= min < max`。启用 `boundary_enforced` 时，六种平移的整条路径都必须处于该高度区间内；垂直升降同时检查当前位置是否在水平允许区域。越界拒绝整步，不截短幅度，继续保存受阻观察和轨迹。初始位姿不会被调整；若其高度在区间外，平移会受阻，需调整配置，观察和转向仍可用。每张相机元数据 JSON 的 `height_limits` 保存上下限、地面参考点、竖直轴和启用状态。这些是配置的高度界限，并非自动检测的天花板，不检查局部低顶或家具。俯视图中纯升降的位置投影重合，实际高度变化记录在轨迹中。

`boundary_preview=True` 控制本地俯视图是否显示边界，关闭显示不会关闭移动限制。黄色线表示场景外轮廓，绿色线表示按 `camera_radius` 向内收缩后的允许移动范围。边界使用完整点云中距地面 -0.1 米至 `trajectory_max_height` 的点进行投影，填满所有封闭的内部空洞，保留与外部连通的凹口和 L 形结构，并选取与初始位置连通的区域；若起点不在轮廓内，则选取最近区域并标注起点异常。家具和内部障碍物不会因此变成碰撞体。

两个入口顶部的 `boundary_cell_size=0.05` 控制网格精度，`boundary_close_radius=0.15` 控制小缺口修补半径，`camera_radius=0.15` 控制内缩距离。设 `boundary_enforced=False` 会关闭水平和高度移动限制，保留按配置显示的边界预览。每个会话的 `boundary.npz` 保存外轮廓掩码、内缩掩码、坐标基、参数、起点标记和 `boundary_enforced` 状态，并用 `outer_only=True` 标明只使用外边界；加载场景时计算一次，`/new` 复用并保存到新目录。

`/new` 创建新的会话目录，清空聊天并恢复初始相机位姿，继续使用同一个已加载场景；旧目录保持原样。`/exit` 退出并关闭 Blender。

普通 Python 运行 agent 和 NumPy / Matplotlib 轨迹绘图；渲染端使用 Blender 自带的 `bpy` 和 NumPy。探索过程不调用 PyTorch、Open3D、VGGT 或 DA3。

## vLLM 启动命令

## Qwen3.8-Flash-Next-FP8 (vLLM 0.31.0)

Uses the native 262,144-token context window without RoPE scaling overrides.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
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

## Qwen3.8-27B-FP8 (vLLM 0.31.0)

Requires `transformers>=5.8.0`. Uses the native 262,144-token context window
without RoPE scaling overrides.

```bash
CUDA_VISIBLE_DEVICES=3 \
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
